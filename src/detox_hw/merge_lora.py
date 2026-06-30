"""Merge a PEFT LoRA adapter into its base model.

verl expects ``actor_rollout_ref.model.path`` to point at a regular
Hugging Face model directory.  Our SFT step saves a LoRA adapter, so PPO
needs a merged checkpoint before it can use SFT as the actor/reference.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(
    adapter_dir: Path,
    out_dir: Path,
    base_name: str = "Qwen/Qwen2.5-0.5B",
    force: bool = False,
) -> None:
    if out_dir.exists():
        if not force:
            model_files = list(out_dir.glob("*.safetensors")) + list(out_dir.glob("pytorch_model*.bin"))
            if (out_dir / "config.json").exists() and model_files:
                print(f"merged checkpoint already exists: {out_dir}")
                return
            raise FileExistsError(
                f"{out_dir} exists but does not look complete; rerun with --force"
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        dtype=torch.float32,
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"saved merged LoRA checkpoint to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    merge_lora(
        adapter_dir=Path(a.adapter_dir),
        out_dir=Path(a.out),
        base_name=a.base,
        force=a.force,
    )


if __name__ == "__main__":
    main()
