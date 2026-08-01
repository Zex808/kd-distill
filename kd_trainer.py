"""
kd_trainer.py — Hinton KD loss + HuggingFace student training.
Supports:
  - Soft-label KD (when teacher logprobs available — OpenAI)
  - Response-only KD (CE on teacher text — Anthropic/Ollama)
  - generation (CausalLM) or classification (SequenceClassification)
  - Optional W&B experiment tracking
"""

import json, math
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import yaml

# Optional W&B
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── Soft-label helpers ────────────────────────────────────────────────────────

def build_sparse_teacher_targets(
    raw_logprobs: list[dict],   # [{token_str: logprob, ...}, ...] — one dict per position
    tokenizer,
    max_len: int,
    top_k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Map API teacher logprobs (teacher tokenizer) → student vocab indices + values.

    Returns:
      indices  : (max_len, top_k)  — student vocab token ids
      logprobs : (max_len, top_k)  — corresponding log probabilities

    ⚠️ Cross-tokenizer caveat: OpenAI uses tiktoken; student models use their own
    BPE/WordPiece. This does best-effort string matching. Works well for GPT-2 family
    students; degrades for DistilBERT (WordPiece). Positions with no match get uniform
    distribution (effectively no KD signal at that position).
    """
    # Build string→id cache once per tokenizer instance
    if not hasattr(tokenizer, "_str_to_id"):
        vocab = tokenizer.get_vocab()
        # Both direct key and decoded form
        tokenizer._str_to_id = {}
        for tok_str, tok_id in vocab.items():
            tokenizer._str_to_id[tok_str] = tok_id
            decoded = tokenizer.convert_tokens_to_string([tok_str])
            tokenizer._str_to_id[decoded] = tok_id

    indices  = torch.zeros(max_len, top_k, dtype=torch.long)
    logprobs = torch.full((max_len, top_k), -1e9)  # ~log(0)

    for t, token_dict in enumerate(raw_logprobs[:max_len]):
        # Sort teacher tokens by logprob desc, take top_k
        sorted_items = sorted(token_dict.items(), key=lambda x: -x[1])[:top_k]
        slot = 0
        for token_str, lp in sorted_items:
            sid = tokenizer._str_to_id.get(token_str)
            if sid is None:
                # Last resort: encode and take first subword
                ids = tokenizer.encode(token_str, add_special_tokens=False)
                sid = ids[0] if ids else None
            if sid is not None:
                indices[t, slot]  = sid
                logprobs[t, slot] = lp
                slot += 1
            if slot >= top_k:
                break

    return indices, logprobs


def sparse_kd_loss(
    student_logits: torch.Tensor,     # (B, T, V)
    teacher_indices: torch.Tensor,    # (B, T, k)
    teacher_logprobs: torch.Tensor,   # (B, T, k)
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Memory-efficient KD loss using only top-k teacher tokens.
    Avoids materializing full (B, T, V) teacher tensor.
    KL computed only over teacher's top-k → approximate but practical.
    """
    T = temperature
    # Gather student logits at teacher token positions
    student_at_k = torch.gather(
        student_logits / T,               # (B, T, V)
        2,                                # vocab dim
        teacher_indices,                  # (B, T, k)
    )  # → (B, T, k)

    # Distributions over top-k
    student_log_soft = F.log_softmax(student_at_k, dim=-1)
    teacher_soft     = F.softmax(teacher_logprobs / T, dim=-1)

    # KL divergence, scaled by T²
    kd = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (T ** 2)
    return kd


def make_collate_fn(tokenizer, max_length: int, top_k: int = 5):
    """
    Returns a DataLoader collate_fn that:
      1. Stacks standard tensors
      2. Parses raw_logprobs JSON strings
      3. Builds sparse teacher target tensors for records that have logprobs
      4. Returns teacher_indices/teacher_logprobs as None when not available
    """
    def collate_fn(batch):
        input_ids     = torch.stack([b["input_ids"]     for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])

        indices_list  = []
        logprobs_list = []
        has_any = False

        for b in batch:
            raw = json.loads(b["raw_logprobs"])
            if raw is not None:
                has_any = True
                idx, lp = build_sparse_teacher_targets(raw, tokenizer, max_length, top_k)
                indices_list.append(idx)
                logprobs_list.append(lp)
            else:
                # Placeholder zeros — won't be used when teacher_indices is None
                indices_list.append(torch.zeros(max_length, top_k, dtype=torch.long))
                logprobs_list.append(torch.full((max_length, top_k), -1e9))

        if has_any:
            teacher_indices  = torch.stack(indices_list)   # (B, T, k)
            teacher_logprobs = torch.stack(logprobs_list)  # (B, T, k)
        else:
            teacher_indices  = None
            teacher_logprobs = None

        return {
            "input_ids":        input_ids,
            "attention_mask":   attention_mask,
            "teacher_indices":  teacher_indices,
            "teacher_logprobs": teacher_logprobs,
        }

    return collate_fn


# ── Dataset ───────────────────────────────────────────────────────────────────

class DistillDataset(Dataset):
    """
    Loads JSONL harvest output: {prompt, response, logprobs}
    Tokenizes prompt+response as training text for generation,
    or prompt-only for classification.
    """

    def __init__(self, path: str, tokenizer, max_length: int, task: str):
        self.records = [
            json.loads(l)
            for l in Path(path).read_text().splitlines()
            if l.strip()
        ]
        self.tok = tokenizer
        self.max_len = max_length
        self.task = task

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        if self.task == "generation":
            # Concat prompt + response; train on full sequence
            text = r["prompt"].strip() + " " + r["response"].strip()
        else:
            # Classification: only the prompt is input
            text = r["prompt"].strip()

        enc = self.tok(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            # Stash raw logprobs string — sparse, only used when not None
            "raw_logprobs": json.dumps(r["logprobs"]),
        }


# ── Loss ──────────────────────────────────────────────────────────────────────

def kd_loss(
    student_logits: torch.Tensor,                         # (B, T, V) or (B, num_labels)
    labels: torch.Tensor,                                 # (B, T) or (B,)
    alpha: float = 0.7,
    temperature: float = 4.0,
    teacher_indices: Optional[torch.Tensor] = None,       # (B, T, k)
    teacher_logprobs_sparse: Optional[torch.Tensor] = None,  # (B, T, k)
) -> torch.Tensor:
    """
    Hinton 2015 KD loss:
      L = α · T² · KL(sparse_student ‖ sparse_teacher) + (1–α) · CE(hard)
    Falls back to pure CE when no teacher logprobs available.
    Uses sparse top-k teacher targets — memory-efficient vs full vocab tensor.
    """
    # Hard CE loss (always)
    if student_logits.dim() == 3:
        ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
    else:
        ce = F.cross_entropy(student_logits, labels)

    if teacher_indices is None or teacher_logprobs_sparse is None:
        return ce  # response-only mode

    kd = sparse_kd_loss(student_logits, teacher_indices, teacher_logprobs_sparse, temperature)
    return alpha * kd + (1.0 - alpha) * ce


# ── Trainer ───────────────────────────────────────────────────────────────────

class KDTrainer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self._setup_model()

    def _setup_model(self):
        sc = self.cfg["student"]
        self.tok = AutoTokenizer.from_pretrained(sc["pretrained"])
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token  # GPT-2 style

        if sc["task"] == "generation":
            self.model = AutoModelForCausalLM.from_pretrained(sc["pretrained"])
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                sc["pretrained"], num_labels=sc.get("num_labels", 2)
            )

        self.model.to(self.device)

        # FP16 scaler — only when CUDA available
        tc = self.cfg["training"]
        self.use_fp16 = tc.get("fp16", False) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None

    def train(
        self,
        progress_cb: Optional[Callable] = None,  # progress_cb(epoch, total_epochs)
        loss_cb: Optional[Callable] = None,        # loss_cb(step, loss_val)
    ) -> list[float]:
        tc = self.cfg["training"]
        dc = self.cfg["data"]
        sc = self.cfg["student"]

        # Optional W&B init
        wc = self.cfg.get("wandb", {})
        use_wandb = wc.get("enabled", False) and _WANDB_AVAILABLE
        if use_wandb:
            wandb.init(
                project=wc.get("project", "kd-distill"),
                name=wc.get("run_name"),
                config={**tc, **sc, "provider": self.cfg["teacher"]["provider"]},
            )

        # Dataset + collate_fn (handles sparse teacher logprob parsing)
        collate = make_collate_fn(self.tok, sc["max_length"])
        ds = DistillDataset(dc["harvest_file"], self.tok, sc["max_length"], sc["task"])
        val_n = max(1, int(len(ds) * dc.get("val_split", 0.1)))
        train_ds, val_ds = random_split(ds, [len(ds) - val_n, val_n])

        train_dl = DataLoader(
            train_ds, batch_size=tc["batch_size"], shuffle=True,
            num_workers=2, pin_memory=True, collate_fn=collate,
        )
        val_dl = DataLoader(
            val_ds, batch_size=tc["batch_size"], shuffle=False,
            num_workers=2, collate_fn=collate,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=tc["lr"], weight_decay=0.01)
        total_steps = math.ceil(len(train_dl) / tc["grad_accum"]) * tc["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=tc["warmup_steps"], num_training_steps=total_steps
        )

        out_dir = Path(tc["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        step = 0
        loss_history: list[float] = []

        for epoch in range(tc["epochs"]):
            self.model.train()
            optimizer.zero_grad()

            for i, batch in enumerate(train_dl):
                ids  = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = ids.clone()
                labels[mask == 0] = -100

                # Move sparse teacher targets to device if present
                t_idx = batch["teacher_indices"].to(self.device)  if batch["teacher_indices"]  is not None else None
                t_lp  = batch["teacher_logprobs"].to(self.device) if batch["teacher_logprobs"] is not None else None

                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    out = self.model(input_ids=ids, attention_mask=mask)
                    loss = kd_loss(
                        out.logits, labels,
                        alpha=tc["alpha"],
                        temperature=tc["temperature"],
                        teacher_indices=t_idx,
                        teacher_logprobs_sparse=t_lp,
                    )
                    loss = loss / tc["grad_accum"]

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % tc["grad_accum"] == 0:
                    if self.scaler:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    if self.scaler:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                    actual_loss = loss.item() * tc["grad_accum"]
                    loss_history.append(actual_loss)
                    if loss_cb:
                        loss_cb(step, actual_loss)
                    if use_wandb:
                        wandb.log({"train/loss": actual_loss, "step": step})

                    if step % tc["save_steps"] == 0:
                        ckpt = out_dir / f"step_{step}"
                        self.model.save_pretrained(ckpt)
                        self.tok.save_pretrained(ckpt)
                        print(f"\nCheckpoint → {ckpt}")

            val_loss = self._validate(val_dl)
            print(f"\nEpoch {epoch+1}/{tc['epochs']} — val_loss: {val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss, "epoch": epoch + 1})
            if progress_cb:
                progress_cb(epoch + 1, tc["epochs"])

        self.model.save_pretrained(out_dir / "final")
        self.tok.save_pretrained(out_dir / "final")
        if use_wandb:
            wandb.finish()
        return loss_history

    @torch.no_grad()
    def _validate(self, loader) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for batch in loader:
            ids  = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = ids.clone()
            labels[mask == 0] = -100
            out = self.model(input_ids=ids, attention_mask=mask, labels=labels)
            total += out.loss.item()
            count += 1
        self.model.train()
        return total / max(count, 1)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Quick inference — generation task only."""
        self.model.eval()
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tok.eos_token_id,
        )
        # Strip input tokens, decode only new tokens
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new_ids, skip_special_tokens=True)


def main():
    """CLI entry point: python kd_trainer.py"""
    trainer = KDTrainer()

    def prog(ep, total): print(f"  Epoch {ep}/{total}")
    def loss_log(step, loss): print(f"\r  step={step} loss={loss:.4f}", end="")

    losses = trainer.train(progress_cb=prog, loss_cb=loss_log)
    print(f"\n✅ Done — final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
