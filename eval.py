"""
eval.py — ROUGE-L and BERTScore evaluation of student vs teacher responses.

Usage:
  python eval.py                            # uses config.yaml defaults
  python eval.py --checkpoint ./checkpoints/final --data data/teacher_data.jsonl
  python eval.py --n 50                     # sample 50 records

Outputs a summary table + saves results to eval_results.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Optional metric imports ───────────────────────────────────────────────────

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠ rouge-score not installed — ROUGE metrics disabled. pip install rouge-score")

try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("⚠ bert-score not installed — BERTScore disabled. pip install bert-score")


# ── Core evaluation ───────────────────────────────────────────────────────────

def load_dataset(path: str, n: int | None = None) -> list[dict]:
    """Load harvest JSONL, optionally cap at n records."""
    records = [
        json.loads(l)
        for l in Path(path).read_text().splitlines()
        if l.strip()
    ]
    return records[:n] if n else records


@torch.no_grad()
def generate_student_responses(
    checkpoint: str,
    prompts: list[str],
    max_new_tokens: int = 200,
    device: str = "cuda",
) -> list[str]:
    """Batch generate responses from student checkpoint."""
    print(f"Loading student from {checkpoint}...")
    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    model.eval()

    responses = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for deterministic eval
            pad_token_id=tok.eos_token_id,
        )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        responses.append(tok.decode(new_ids, skip_special_tokens=True))

    return responses


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    if not ROUGE_AVAILABLE:
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k] += scores[k].fmeasure
    n = len(predictions)
    return {k: round(v / n, 4) for k, v in agg.items()}


def compute_bert_score(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
) -> dict:
    """BERTScore P/R/F1 — semantic similarity metric."""
    if not BERT_SCORE_AVAILABLE:
        return {}
    P, R, F1 = bert_score_fn(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_p":  round(P.mean().item(), 4),
        "bertscore_r":  round(R.mean().item(), 4),
        "bertscore_f1": round(F1.mean().item(), 4),
    }


def run_eval(
    checkpoint: str,
    data_path: str,
    n: int | None = None,
    output_path: str = "eval_results.jsonl",
    device: str | None = None,
) -> dict:
    """Full eval pipeline. Returns aggregate metrics dict."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    records = load_dataset(data_path, n)
    prompts   = [r["prompt"] for r in records]
    teacher_responses = [r["response"] for r in records]

    print(f"Evaluating {len(records)} records on {device}...")
    student_responses = generate_student_responses(checkpoint, prompts, device=device)

    rouge  = compute_rouge(student_responses, teacher_responses)
    bscore = compute_bert_score(student_responses, teacher_responses)
    metrics = {**rouge, **bscore}

    # Save per-record results
    with open(output_path, "w") as f:
        for prompt, teacher, student in zip(prompts, teacher_responses, student_responses):
            # Per-record ROUGE-L for inspection
            per = {}
            if ROUGE_AVAILABLE:
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                per["rougeL"] = round(scorer.score(teacher, student)["rougeL"].fmeasure, 4)
            f.write(json.dumps({
                "prompt": prompt,
                "teacher": teacher,
                "student": student,
                **per,
            }) + "\n")

    print(f"\n{'─'*40}")
    print(f"  Eval results ({len(records)} samples)")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")
    print(f"{'─'*40}")
    print(f"  Per-record → {output_path}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    default_ckpt = str(Path(cfg["training"]["output_dir"]) / "final")
    default_data = cfg["data"]["harvest_file"]

    parser = argparse.ArgumentParser(description="Evaluate student vs teacher")
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--data",       default=default_data)
    parser.add_argument("--n",          type=int, default=None, help="Max records to eval")
    parser.add_argument("--output",     default="eval_results.jsonl")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)

    run_eval(args.checkpoint, args.data, args.n, args.output, args.device)


if __name__ == "__main__":
    main()
