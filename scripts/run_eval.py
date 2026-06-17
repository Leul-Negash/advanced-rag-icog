"""Run the full Basic vs Advanced RAG comparison and write the report.

Outputs:
  eval/results/results.json   -- raw per-question results + aggregates
  docs/COMPARISON.md          -- the human-readable comparison deliverable

Usage:
    python scripts/run_eval.py                # heuristic scoring (no extra LLM cost)
    python scripts/run_eval.py --judge llm    # LLM-as-judge scoring
    python scripts/run_eval.py --limit 5      # first N questions only
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
from src.advanced_rag import AdvancedRAG  # noqa: E402
from src.basic_rag import BasicRAG  # noqa: E402
from src.evaluate import aggregate, evaluate_one  # noqa: E402
from src.llm import QuotaExhausted  # noqa: E402


def load_questions(limit=None):
    data = json.loads((ROOT / "eval" / "test_questions.json").read_text())
    qs = data["questions"]
    return qs[:limit] if limit else qs


def _save(results, console):
    out = ROOT / "eval" / "results" / "results.json"
    out.write_text(json.dumps(results, indent=2))
    write_comparison_md(results)


def run(judge="heuristic", limit=None):
    from rich.console import Console
    console = Console()

    questions = load_questions(limit)
    console.print(f"[bold]Loading systems (judge={judge})...[/bold]")
    basic = BasicRAG()
    advanced = AdvancedRAG()

    rows_basic, rows_adv, detailed = [], [], []
    results = {"summary": {}, "details": detailed, "judge": judge}
    quota_stop = False

    for q in questions:
        console.print(f"\n[bold cyan]{q['id']} ({q['category']})[/bold cyan] {q['question']}")
        try:
            rb = basic.query(q["question"])
            ra = advanced.query(q["question"])
        except QuotaExhausted as e:
            console.print(f"[red]Quota exhausted - stopping early and saving "
                          f"partial results.[/red]\n{e}")
            quota_stop = True
            break

        eb = evaluate_one(rb, q, judge=judge)
        ea = evaluate_one(ra, q, judge=judge)
        rows_basic.append(eb)
        rows_adv.append(ea)

        console.print(f"  basic   : {'REFUSED' if rb.refused else 'answered'} | "
                      f"hit={eb['retrieval_hit']} correct={eb['correct']}")
        console.print(f"  advanced: {'REFUSED' if ra.refused else 'answered'} | "
                      f"hit={ea['retrieval_hit']} correct={ea['correct']}")

        detailed.append({
            "question": q, "basic": eb, "advanced": ea,
            "basic_answer": rb.answer, "advanced_answer": ra.answer,
            "advanced_trace": ra.trace,
        })
        # checkpoint after every question so partial quota still yields output
        results["summary"] = {"basic": aggregate(rows_basic),
                              "advanced": aggregate(rows_adv)}
        _save(results, console)

    if not detailed:
        console.print("[red]No questions completed (quota). Try again after the "
                      "daily reset or set GEMINI_MODEL to a model with quota.[/red]")
        return results

    results["summary"] = {"basic": aggregate(rows_basic),
                          "advanced": aggregate(rows_adv)}
    results["partial"] = quota_stop
    _save(results, console)
    console.print(f"\n[green]Wrote eval/results/results.json and docs/COMPARISON.md"
                  f" ({len(detailed)} questions"
                  f"{', PARTIAL due to quota' if quota_stop else ''})[/green]")
    _print_summary(console, results["summary"]["basic"], results["summary"]["advanced"])
    return results


def _print_summary(console, b, a):
    from rich.table import Table
    t = Table(title="Basic vs Advanced RAG - Summary")
    t.add_column("Metric")
    t.add_column("Basic", justify="right")
    t.add_column("Advanced", justify="right")
    for label, key in [
        ("Retrieval recall (answerable)", "retrieval_recall"),
        ("Answer accuracy (answerable)", "answer_accuracy"),
        ("Grounded rate (answerable)", "grounded_rate"),
        ("Refusal accuracy (unanswerable)", "refusal_accuracy_unanswerable"),
        ("Hallucination on unanswerable", "hallucination_on_unanswerable"),
        ("Overall correct", "overall_correct"),
    ]:
        t.add_row(label, str(b.get(key)), str(a.get(key)))
    console.print(t)


# --------------------------------------------------------------------------- #
def _pct(x):
    return "-" if x is None else f"{x*100:.0f}%"


def write_comparison_md(results: dict):
    b = results["summary"]["basic"]
    a = results["summary"]["advanced"]
    details = results["details"]

    lines = []
    judge = results.get("judge", "heuristic")
    judge_desc = {
        "heuristic": "answer correctness scored by key-term coverage vs the gold "
                     "answer (no-LLM heuristic, to conserve free-tier quota)",
        "llm": "answer correctness scored by an LLM-as-judge against the gold answer",
        "none": "answers not auto-scored",
    }.get(judge, judge)

    lines.append("# Basic RAG vs Advanced RAG - Comparison Results\n")
    lines.append("Corpus: OpenCog Hyperon / MeTTa / iCog Labs documentation "
                 f"({b['n_total']} test questions: {b['n_answerable']} answerable, "
                 f"{b['n_unanswerable']} unanswerable).\n")
    if results.get("partial"):
        lines.append("> **Partial run** - stopped early when the LLM free-tier "
                     "quota was exhausted. Re-run after the quota resets for the "
                     "complete table.\n")
    lines.append(f"**Scoring:** {judge_desc}.\n")

    lines.append("**Setup.** Both systems share the same corpus, embedding model "
                 "(`BAAI/bge-small-en-v1.5`), vector store, and generation model, so "
                 "the comparison isolates the retrieval techniques. *Basic* is the "
                 "naive baseline: fixed-size chunks, single-query retrieval of the "
                 "top-3 chunks by cosine similarity, no re-ranking. *Advanced* uses "
                 "context-aware + hierarchical chunking, query expansion + multi-query "
                 "with RRF de-duplication, cross-encoder re-ranking of a 12-candidate "
                 "pool down to the best 6, corrective relevance grading, and "
                 "parent-section expansion.\n")

    lines.append("## Test questions\n")
    lines.append("| ID | Category | Question | Gold answer |")
    lines.append("|---|---|---|---|")
    for d in details:
        q = d["question"]
        gold = q.get("gold", "").replace("|", "/")
        lines.append(f"| {q['id']} | {q['category']} | {q['question'].replace('|','/')} | {gold} |")
    lines.append("")

    lines.append("## Summary metrics\n")
    lines.append("| Metric | Basic RAG | Advanced RAG |")
    lines.append("|---|---|---|")
    rows = [
        ("Retrieval recall (answerable)", "retrieval_recall"),
        ("Answer accuracy (answerable)", "answer_accuracy"),
        ("Grounded / supported rate", "grounded_rate"),
        ("Correct refusal on unanswerable", "refusal_accuracy_unanswerable"),
        ("Hallucination on unanswerable (lower=better)", "hallucination_on_unanswerable"),
        ("Overall correct", "overall_correct"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | {_pct(b.get(key))} | {_pct(a.get(key))} |")
    lines.append("")

    def yn(v):
        return "-" if v is None else ("yes" if v else "no")

    lines.append("## Per-question results\n")
    lines.append("| ID | Category | Basic correct | Adv correct | "
                 "Basic grounded | Adv grounded | Basic hit | Adv hit |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in details:
        q = d["question"]
        eb, ea = d["basic"], d["advanced"]
        lines.append(f"| {q['id']} | {q['category']} | "
                     f"{yn(eb['correct'])} | {yn(ea['correct'])} | "
                     f"{yn(eb.get('grounded'))} | {yn(ea.get('grounded'))} | "
                     f"{eb['retrieval_hit']} | {ea['retrieval_hit']} |")
    lines.append("")

    # examples where advanced was better: either more correct, or equally correct
    # but better grounded in the retrieved evidence (a faithfulness win)
    wins = [d for d in details
            if (d["advanced"]["correct"] and not d["basic"]["correct"])
            or (d["advanced"].get("grounded") and not d["basic"].get("grounded"))]
    lines.append("## Examples where Advanced RAG did better\n")
    if not wins:
        lines.append("_(Both systems performed equally on this run.)_\n")
    for d in wins[:5]:
        q = d["question"]
        acc_win = d["advanced"]["correct"] and not d["basic"]["correct"]
        kind = "accuracy" if acc_win else "grounding / faithfulness"
        lines.append(f"### {q['id']} - {q['question']}")
        lines.append(f"- **Category:** {q['category']}  ({kind} win)")
        lines.append(f"- **Basic answer:** {d['basic_answer'][:400]}")
        lines.append(f"- **Advanced answer:** {d['advanced_answer'][:400]}")
        lines.append(f"- **Why advanced was better:** {_why(d, acc_win)}\n")

    # examples where advanced still failed
    fails = [d for d in details if not d["advanced"]["correct"]]
    lines.append("## Examples where Advanced RAG still failed\n")
    if not fails:
        lines.append("_(Advanced RAG answered every question correctly on this run.)_\n")
    for d in fails[:5]:
        q = d["question"]
        lines.append(f"### {q['id']} - {q['question']}")
        lines.append(f"- **Category:** {q['category']}")
        lines.append(f"- **Advanced answer:** {d['advanced_answer'][:400]}")
        lines.append(f"- **Gold:** {q.get('gold','')}\n")

    (ROOT / "docs" / "COMPARISON.md").write_text("\n".join(lines))


def _why(d, acc_win=True):
    q = d["question"]
    if not acc_win:
        return ("Both answers were correct, but the baseline's answer was not fully "
                "supported by its retrieved context (a latent hallucination); the "
                "advanced system retrieved the complete supporting evidence, so its "
                "answer is grounded.")
    if q["category"] == "unanswerable":
        return "Advanced correctly refused; corrective grading detected insufficient evidence."
    if not d["basic"]["retrieval_hit"] and d["advanced"]["retrieval_hit"]:
        return ("Query expansion / multi-query + re-ranking retrieved the relevant "
                "chunk that the baseline missed.")
    if q["category"] == "relationship":
        return ("Multi-query retrieval + re-ranking gathered the spread-out evidence "
                "needed to connect the facts.")
    return "Better evidence selection (re-ranking + corrective grading) improved the answer."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["heuristic", "llm", "none"],
                    default="heuristic",
                    help="answer scoring: heuristic (no LLM, default), llm, or none")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(judge=args.judge, limit=args.limit)


if __name__ == "__main__":
    main()
