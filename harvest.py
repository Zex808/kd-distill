"""
harvest.py — Pull teacher responses + logprobs from API or Ollama.
Output: JSONL where each line = {prompt, response, logprobs}
logprobs is None for Anthropic/Ollama (response-only distillation)
logprobs is list[dict[token, logprob]] for OpenAI (true soft-label KD)
"""

import json, os, asyncio
from pathlib import Path
from typing import Callable, Optional
import httpx, yaml


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts(path: str) -> list[str]:
    """Accept .jsonl ({prompt: ...}) or plain .txt (one per line)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    if p.suffix == ".jsonl":
        return [json.loads(l)["prompt"] for l in lines]
    return lines


# ── Provider implementations ─────────────────────────────────────────────────

async def harvest_openai(
    prompts: list[str], cfg: dict, out_path: str,
    progress_cb: Optional[Callable] = None
):
    """OpenAI: captures top_logprobs → real soft-label KD possible."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv(cfg["teacher"]["api_key_env"]))
    tc = cfg["teacher"]

    with open(out_path, "a") as f:
        for i, prompt in enumerate(prompts):
            resp = await client.chat.completions.create(
                model=tc["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=tc["temperature"],
                logprobs=tc["logprobs"],
                top_logprobs=tc["top_logprobs"] if tc["logprobs"] else None,
                max_tokens=cfg["student"]["max_length"],
            )
            choice = resp.choices[0]

            # Build token-level logprob dict: [{token: logprob, ...}, ...]
            lp = None
            if choice.logprobs and choice.logprobs.content:
                lp = [
                    {t.token: t.logprob for t in step.top_logprobs}
                    for step in choice.logprobs.content
                ]

            f.write(json.dumps({
                "prompt": prompt,
                "response": choice.message.content,
                "logprobs": lp,
            }) + "\n")

            if progress_cb:
                progress_cb(i + 1, len(prompts))


async def harvest_anthropic(
    prompts: list[str], cfg: dict, out_path: str,
    progress_cb: Optional[Callable] = None
):
    """Anthropic: response-only KD (no logprobs API)."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.getenv(cfg["teacher"]["api_key_env"]))
    tc = cfg["teacher"]

    with open(out_path, "a") as f:
        for i, prompt in enumerate(prompts):
            msg = await client.messages.create(
                model=tc["model"],
                max_tokens=cfg["student"]["max_length"],
                messages=[{"role": "user", "content": prompt}],
            )
            f.write(json.dumps({
                "prompt": prompt,
                "response": msg.content[0].text,
                "logprobs": None,
            }) + "\n")

            if progress_cb:
                progress_cb(i + 1, len(prompts))


async def harvest_ollama(
    prompts: list[str], cfg: dict, out_path: str,
    progress_cb: Optional[Callable] = None
):
    """Ollama: local teacher, response-only KD."""
    tc = cfg["teacher"]

    async with httpx.AsyncClient(timeout=180) as client:
        for i, prompt in enumerate(prompts):
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": tc["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": tc["temperature"],
                        "num_predict": cfg["student"]["max_length"],
                    },
                },
            )
            data = resp.json()
            with open(out_path, "a") as f:
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": data["response"],
                    "logprobs": None,
                }) + "\n")

            if progress_cb:
                progress_cb(i + 1, len(prompts))


HARVESTERS = {
    "openai": harvest_openai,
    "anthropic": harvest_anthropic,
    "ollama": harvest_ollama,
}


# ── HuggingFace dataset loader ────────────────────────────────────────────────

def load_prompts_from_hf(cfg: dict) -> list[str]:
    """
    Pull prompts from a HuggingFace dataset instead of a local file.
    Requires: datasets >= 2.19.0
    Example config:
      hf_dataset: tatsu-lab/alpaca
      hf_split: train
      hf_prompt_field: instruction
      hf_max_rows: 1000
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets  # required for HF dataset support")

    dc = cfg["data"]
    print(f"Loading HF dataset: {dc['hf_dataset']} ({dc.get('hf_split', 'train')})...")
    ds = hf_load(dc["hf_dataset"], split=dc.get("hf_split", "train"))

    field = dc.get("hf_prompt_field", "prompt")
    max_rows = dc.get("hf_max_rows", 1000)

    if field not in ds.column_names:
        raise ValueError(f"Field '{field}' not found. Available: {ds.column_names}")

    prompts = [str(row[field]) for row in ds.select(range(min(max_rows, len(ds))))]
    print(f"  Loaded {len(prompts)} prompts from {dc['hf_dataset']}")
    return prompts


# ── Public entry point ────────────────────────────────────────────────────────

def run_harvest(
    config_path="config.yaml",
    progress_cb: Optional[Callable] = None,
) -> int:
    cfg = load_config(config_path)
    dc = cfg["data"]

    # HF dataset takes priority over local file if configured
    if dc.get("hf_dataset"):
        prompts = load_prompts_from_hf(cfg)
        # Write to local file so harvest step re-uses same code path
        Path(dc["prompts_file"]).parent.mkdir(parents=True, exist_ok=True)
        with open(dc["prompts_file"], "w") as f:
            for p in prompts:
                f.write(json.dumps({"prompt": p}) + "\n")
    else:
        prompts = load_prompts(dc["prompts_file"])

    Path(dc["harvest_file"]).parent.mkdir(parents=True, exist_ok=True)

    provider = cfg["teacher"]["provider"]
    if provider not in HARVESTERS:
        raise ValueError(f"Unknown provider: {provider}. Choose: {list(HARVESTERS)}")

    asyncio.run(HARVESTERS[provider](prompts, cfg, dc["harvest_file"], progress_cb))
    return len(prompts)


def main():
    """CLI entry point: python harvest.py"""
    def prog(i, total):
        print(f"\rHarvesting {i}/{total}", end="", flush=True)

    n = run_harvest(progress_cb=prog)
    print(f"\n✅ Done — {n} records written")


if __name__ == "__main__":
    main()
