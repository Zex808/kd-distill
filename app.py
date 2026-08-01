"""
app.py — Gradio UI for KD Pipeline.
Run: python app.py  →  http://localhost:7860

Tabs:
  1. Harvest  — pull teacher responses from API/Ollama
  2. Train    — distill into student on local GPU
  3. Evaluate — side-by-side student vs teacher
  4. Config   — live YAML editor
"""

import gradio as gr
import json, os, yaml, asyncio, threading
from pathlib import Path

from harvest import run_harvest, load_prompts, HARVESTERS
from kd_trainer import KDTrainer
from eval import run_eval

# ── Global state ──────────────────────────────────────────────────────────────
_trainer: KDTrainer = None
_train_thread: threading.Thread = None
_loss_log: list = []  # [(step, loss)]


def _load_cfg() -> dict:
    return yaml.safe_load(open("config.yaml"))

def _save_cfg(cfg: dict):
    with open("config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ── Tab 1: Harvest ─────────────────────────────────────────────────────────────

def do_harvest(provider, model, api_key, prompts_text, temperature, enable_logprobs, progress=gr.Progress()):
    """Write prompts → JSONL, patch config, run harvest."""
    Path("data").mkdir(exist_ok=True)

    # Write prompts file
    with open("data/prompts.jsonl", "w") as f:
        for line in prompts_text.strip().splitlines():
            line = line.strip()
            if line:
                f.write(json.dumps({"prompt": line}) + "\n")

    # Set env var for API key
    env_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "ollama": ""}
    if env_map[provider]:
        os.environ[env_map[provider]] = api_key

    # Patch config
    cfg = _load_cfg()
    cfg["teacher"].update({
        "provider": provider,
        "model": model,
        "temperature": float(temperature),
        "logprobs": enable_logprobs,
        "api_key_env": env_map.get(provider, "OPENAI_API_KEY"),
    })
    _save_cfg(cfg)

    try:
        def _prog(i, total):
            progress(i / total, desc=f"{i}/{total} prompts")

        n = run_harvest(progress_cb=_prog)
        lines = Path(cfg["data"]["harvest_file"]).read_text().splitlines()
        has_logprobs = sum(1 for l in lines if json.loads(l).get("logprobs"))
        return (
            f"✅ Harvested {n} records\n"
            f"📊 Soft-label logprobs: {has_logprobs}/{n} records\n"
            f"💾 Saved → {cfg['data']['harvest_file']}"
        )
    except Exception as e:
        return f"❌ Error: {e}"


# ── Tab 2: Train ───────────────────────────────────────────────────────────────

def do_train(
    pretrained, task, alpha, kd_temp,
    epochs, batch_size, lr, fp16,
    progress=gr.Progress()
):
    global _trainer, _loss_log
    _loss_log = []

    # Patch config
    cfg = _load_cfg()
    cfg["student"].update({"pretrained": pretrained, "task": task})
    cfg["training"].update({
        "alpha": float(alpha),
        "temperature": float(kd_temp),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "fp16": fp16,
    })
    _save_cfg(cfg)

    try:
        _trainer = KDTrainer()

        def _prog(ep, total):
            progress(ep / total, desc=f"Epoch {ep}/{total}")

        def _loss_cb(step, loss):
            _loss_log.append({"step": step, "loss": round(loss, 5)})

        losses = _trainer.train(progress_cb=_prog, loss_cb=_loss_cb)

        # Build plot data
        plot_data = [[i, l] for i, l in enumerate(losses)]
        return (
            f"✅ Training done! Final loss: {losses[-1]:.4f}\n"
            f"📁 Saved → {cfg['training']['output_dir']}/final",
            plot_data,
        )
    except Exception as e:
        return f"❌ {e}", []


# ── Tab 3: Evaluate ────────────────────────────────────────────────────────────

def eval_student(prompt):
    if _trainer is None:
        return "⚠️ Train a model first (or load a checkpoint)."
    try:
        return _trainer.generate(prompt)
    except Exception as e:
        return f"❌ {e}"


async def eval_teacher_api(prompt, provider, model, api_key):
    """Live teacher call for comparison — mirrors harvest logic."""
    cfg = _load_cfg()
    env_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    if env_map.get(provider):
        os.environ[env_map[provider]] = api_key

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as tmp:
        tmp_path = tmp.name

    cfg["teacher"].update({"provider": provider, "model": model})

    # Write single prompt, harvest, read result
    Path("data/tmp_prompt.jsonl").write_text(json.dumps({"prompt": prompt}) + "\n")
    cfg["data"]["prompts_file"] = "data/tmp_prompt.jsonl"
    cfg["data"]["harvest_file"] = tmp_path

    try:
        await HARVESTERS[provider]([prompt], cfg, tmp_path)
        record = json.loads(Path(tmp_path).read_text().splitlines()[0])
        return record["response"]
    except Exception as e:
        return f"❌ {e}"


def eval_teacher(prompt, provider, model, api_key):
    return asyncio.run(eval_teacher_api(prompt, provider, model, api_key))


def do_batch_eval(n_samples, progress=gr.Progress()):
    """Run ROUGE + BERTScore eval on harvest dataset vs student checkpoint."""
    cfg = _load_cfg()
    ckpt = str(Path(cfg["training"]["output_dir"]) / "final")
    data = cfg["data"]["harvest_file"]

    if not Path(ckpt).exists():
        return "⚠️ No checkpoint found. Train a model first.", ""
    if not Path(data).exists():
        return "⚠️ No harvest data found. Run Harvest first.", ""

    try:
        progress(0.1, desc="Loading student checkpoint...")
        n = int(n_samples) if n_samples else None
        metrics = run_eval(ckpt, data, n=n, output_path="eval_results.jsonl")

        progress(1.0, desc="Done")
        lines = [f"  {k:<22} {v:.4f}" for k, v in metrics.items()]
        summary = "📊 Aggregate Metrics\n" + "─" * 35 + "\n" + "\n".join(lines)
        return summary, "✅ Per-record results → eval_results.jsonl"
    except Exception as e:
        return f"❌ {e}", ""


# ── Tab 4: Config ──────────────────────────────────────────────────────────────

def load_config_text():
    return Path("config.yaml").read_text()

def save_config_text(text):
    try:
        yaml.safe_load(text)  # validate
        Path("config.yaml").write_text(text)
        return "✅ Config saved"
    except yaml.YAMLError as e:
        return f"❌ YAML error: {e}"


# ── UI Layout ─────────────────────────────────────────────────────────────────

CSS = """
.tab-nav button { font-size: 15px !important; }
.status-box textarea { font-family: monospace; }
"""

with gr.Blocks(title="KD Pipeline", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        "# 🎓 Knowledge Distillation Pipeline\n"
        "**API / Ollama Teacher → Small Local Student**  |  "
        "[harvest logits] → [train w/ KD loss] → [eval]"
    )

    # ── Step 1: Harvest ────────────────────────────────────────────────────────
    with gr.Tab("1 · Harvest"):
        gr.Markdown("### Pull teacher responses (+ logprobs where available)")
        with gr.Row():
            h_provider = gr.Dropdown(
                ["openai", "anthropic", "ollama"], value="openai", label="Provider", scale=1
            )
            h_model = gr.Textbox(value="gpt-4o-mini", label="Model ID", scale=2)
        h_key = gr.Textbox(
            placeholder="sk-... (not needed for Ollama)",
            label="API Key", type="password"
        )
        h_prompts = gr.Textbox(
            placeholder="One prompt per line:\nExplain transformers in simple terms.\nWhat is backpropagation?",
            label="Prompts", lines=8
        )
        with gr.Row():
            h_temp = gr.Slider(0.1, 2.0, value=1.0, label="Temperature (higher = softer targets)")
            h_logprobs = gr.Checkbox(value=True, label="Capture logprobs (OpenAI only)")
        h_btn = gr.Button("🌾 Harvest", variant="primary")
        h_out = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")

        h_btn.click(
            do_harvest,
            inputs=[h_provider, h_model, h_key, h_prompts, h_temp, h_logprobs],
            outputs=h_out,
        )

    # ── Step 2: Train ──────────────────────────────────────────────────────────
    with gr.Tab("2 · Train"):
        gr.Markdown("### Distill into student model on local GPU")
        with gr.Row():
            t_model = gr.Dropdown(
                ["distilgpt2", "gpt2", "gpt2-medium",
                 "distilbert-base-uncased", "microsoft/phi-2"],
                value="distilgpt2", label="Student (HF model ID)", scale=2
            )
            t_task = gr.Dropdown(
                ["generation", "classification"], value="generation", label="Task", scale=1
            )
        with gr.Row():
            t_alpha = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                label="α — KD loss weight  (1-α = CE weight)")
            t_temp  = gr.Slider(1.0, 10.0, value=4.0, step=0.5,
                                label="KD Temperature  (Hinton: 4–8)")
        with gr.Row():
            t_epochs = gr.Slider(1, 20, value=3, step=1, label="Epochs")
            t_batch  = gr.Slider(1, 32, value=4, step=1, label="Batch Size")
        with gr.Row():
            t_lr   = gr.Number(value=2e-5, label="Learning Rate")
            t_fp16 = gr.Checkbox(value=True, label="FP16 (GPU)")
        t_btn = gr.Button("🏋️ Train", variant="primary")
        t_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
        t_plot = gr.LinePlot(
            x="step", y="loss",
            title="Training Loss",
            x_title="Step", y_title="Loss",
        )

        t_btn.click(
            do_train,
            inputs=[t_model, t_task, t_alpha, t_temp, t_epochs, t_batch, t_lr, t_fp16],
            outputs=[t_status, t_plot],
        )

    # ── Step 3: Evaluate ───────────────────────────────────────────────────────
    with gr.Tab("3 · Evaluate"):
        gr.Markdown("### Student vs Teacher side-by-side")
        e_prompt = gr.Textbox(label="Prompt", placeholder="Enter a test prompt...")
        with gr.Row():
            e_provider = gr.Dropdown(["openai", "anthropic", "ollama"], value="openai",
                                     label="Teacher Provider", scale=1)
            e_model = gr.Textbox(value="gpt-4o-mini", label="Teacher Model", scale=2)
        e_key = gr.Textbox(placeholder="API Key", label="API Key", type="password")

        with gr.Row():
            e_student_btn = gr.Button("▶ Student", variant="primary")
            e_teacher_btn = gr.Button("▶ Teacher (live API)", variant="secondary")

        with gr.Row():
            e_student_out = gr.Textbox(label="🎓 Student", lines=6)
            e_teacher_out = gr.Textbox(label="👨‍🏫 Teacher", lines=6)

        e_student_btn.click(eval_student, inputs=e_prompt, outputs=e_student_out)
        e_teacher_btn.click(
            eval_teacher,
            inputs=[e_prompt, e_provider, e_model, e_key],
            outputs=e_teacher_out,
        )

        gr.Markdown("---\n### Batch Eval — ROUGE-L + BERTScore")
        gr.Markdown(
            "_Requires `pip install rouge-score bert-score`. "
            "Scores student checkpoint against teacher responses in harvest file._"
        )
        with gr.Row():
            e_n = gr.Number(value=50, label="# Records to eval (blank = all)", precision=0)
            e_batch_btn = gr.Button("📊 Run Batch Eval", variant="secondary")
        e_metrics_out = gr.Textbox(label="Metrics", interactive=False, lines=8,
                                   elem_classes="status-box")
        e_file_out = gr.Textbox(label="Output", interactive=False)
        e_batch_btn.click(do_batch_eval, inputs=e_n, outputs=[e_metrics_out, e_file_out])

    # ── Step 4: Config ──────────────────────────────────────────────────────────
    with gr.Tab("⚙ Config"):
        gr.Markdown("### Live YAML editor — changes apply on next run")
        cfg_load_btn = gr.Button("↻ Load current config")
        cfg_text = gr.Code(language="yaml", label="config.yaml", lines=30)
        cfg_save_btn = gr.Button("💾 Save Config", variant="primary")
        cfg_status = gr.Textbox(label="Status", interactive=False)

        cfg_load_btn.click(load_config_text, outputs=cfg_text)
        cfg_save_btn.click(save_config_text, inputs=cfg_text, outputs=cfg_status)
        demo.load(load_config_text, outputs=cfg_text)  # auto-load on start


def main():
    demo.launch(server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
