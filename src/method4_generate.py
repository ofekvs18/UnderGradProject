"""
method4_generate.py — LLM inference for biomarker formula generation (issue #21).

Loads Med-Gemma 4B IT and runs all prompt configurations, saving raw outputs to
results/method4_llm/raw_outputs.json. Designed to run on a GPU cluster node.

Usage:
    python src/method4_generate.py                         # all configs, 4 repeats
    python src/method4_generate.py --config blind_temp0.7  # single config
    python src/method4_generate.py --dry-run               # preview prompts, no inference
    python src/method4_generate.py --repeats 2             # fewer repeats (testing)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from method4_prompts import get_all_prompt_configs

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID     = "google/medgemma-4b-it"
OUTPUT_DIR   = Path("results/method4_llm")
OUTPUT_FILE  = OUTPUT_DIR / "raw_outputs.json"
DEFAULT_REPEATS  = 4
MAX_NEW_TOKENS   = 1024
DO_SAMPLE        = True  # needed for temperature > 0


# ── Dry-run: preview prompts without loading the model ────────────────────────
def run_dry(configs: list[dict]) -> None:
    print(f"DRY RUN — {len(configs)} config(s), no inference.\n")
    for cfg in configs:
        print(f"{'='*70}")
        print(f"Config : {cfg['name']}")
        print(f"Strategy    : {cfg['strategy']}")
        print(f"Temperature : {cfg['temperature']}")
        print(f"n_formulas  : {cfg['n_formulas']}")
        print(f"chain_of_thought: {cfg['chain_of_thought']}")
        print(f"\n--- PROMPT ---\n{cfg['prompt']}\n")


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    """Load Med-Gemma 4B IT in bfloat16 with automatic device placement."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install with: pip install transformers torch accelerate")
        sys.exit(1)

    print(f"[{_ts()}] Loading tokenizer from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"[{_ts()}] Loading model (bfloat16, device_map=auto) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    device_info = {k: str(v) for k, v in model.hf_device_map.items()} if hasattr(model, "hf_device_map") else "N/A"
    print(f"[{_ts()}] Model loaded. Device map: {device_info}")
    return model, tokenizer


# ── Single inference call ──────────────────────────────────────────────────────
def generate_one(model, tokenizer, prompt: str, temperature: float) -> tuple[str, float]:
    """
    Run one inference call. Returns (raw_text, elapsed_seconds).
    Uses chat template formatting expected by instruction-tuned models.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs.shape[-1]:]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return raw_text, elapsed


# ── Main inference loop ────────────────────────────────────────────────────────
def run_inference(configs: list[dict], repeats: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load any existing results so we can append (resume-friendly)
    existing: list[dict] = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        print(f"[{_ts()}] Resuming — {len(existing)} existing result(s) found.")

    results = list(existing)
    model, tokenizer = load_model()

    total = len(configs) * repeats
    done  = 0

    for cfg in configs:
        for repeat_idx in range(repeats):
            done += 1
            print(f"[{_ts()}] [{done}/{total}] config={cfg['name']}  repeat={repeat_idx + 1}/{repeats}")

            try:
                raw_text, elapsed = generate_one(
                    model, tokenizer, cfg["prompt"], cfg["temperature"]
                )
                status = "ok"
                error  = None
            except Exception as exc:
                raw_text = ""
                elapsed  = 0.0
                status   = "error"
                error    = str(exc)
                print(f"  [WARN] Inference failed: {exc}")

            entry = {
                "run_id":         f"{cfg['name']}_r{repeat_idx}",
                "config_name":    cfg["name"],
                "strategy":       cfg["strategy"],
                "temperature":    cfg["temperature"],
                "n_formulas":     cfg["n_formulas"],
                "chain_of_thought": cfg["chain_of_thought"],
                "repeat_index":   repeat_idx,
                "prompt":         cfg["prompt"],
                "raw_text":       raw_text,
                "elapsed_sec":    round(elapsed, 2),
                "status":         status,
                "error":          error,
                "timestamp":      datetime.utcnow().isoformat() + "Z",
                "model_id":       MODEL_ID,
            }
            results.append(entry)

            # Write after every call — don't lose work if cluster job is killed
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

            if status == "ok":
                preview = raw_text[:120].replace("\n", " ")
                print(f"  elapsed={elapsed:.1f}s  preview: {preview!r}")

    ok_count  = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")
    print(f"\n[{_ts()}] Done. {ok_count} ok / {err_count} errors → {OUTPUT_FILE}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.utcnow().strftime("%H:%M:%S")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Med-Gemma 4B inference for RA biomarker formula generation."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Run a single named config (e.g. blind_temp0.7). Default: all configs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all prompt configs and exit without loading the model.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of repeat inference calls per config (default: {DEFAULT_REPEATS}).",
    )
    args = parser.parse_args()

    all_configs = get_all_prompt_configs()

    if args.config:
        configs = [c for c in all_configs if c["name"] == args.config]
        if not configs:
            valid = [c["name"] for c in all_configs]
            print(f"[ERROR] Unknown config '{args.config}'. Valid options: {valid}")
            sys.exit(1)
    else:
        configs = all_configs

    if args.dry_run:
        run_dry(configs)
        return

    print(f"[{_ts()}] Starting inference: {len(configs)} config(s) × {args.repeats} repeat(s) = {len(configs) * args.repeats} calls")
    run_inference(configs, args.repeats)


if __name__ == "__main__":
    main()
