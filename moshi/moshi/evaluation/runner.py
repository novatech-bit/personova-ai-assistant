# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Full-Duplex-Bench evaluation runner for PersonaPlex/Moshi.

Runs the model on a dataset of evaluation samples and computes metrics for
the four Full-Duplex-Bench categories: pause handling, backchanneling,
smooth turn-taking, and user interruption.

Usage:
    python -m moshi.evaluation.runner \\
        --dataset-dir /path/to/fullduplexbench/dataset \\
        --output-dir /path/to/results \\
        --categories pause_handling turn_taking interruption

Dataset structure (Full-Duplex-Bench v1/v1.5 format):
    dataset_dir/
        pause_handling/
            metadata.json
            audio/
                sample_001.wav
                ...
        backchannel/
            ...
        turn_taking/
            ...
        interruption/
            ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch

from ..models import loaders
from ..pipeline import VoiceToVoicePipeline
from .metrics import (
    FullDuplexBenchResult,
    evaluate_pause_handling,
    evaluate_turn_taking,
    evaluate_interruption,
)


CATEGORY_MAP = {
    "pause_handling": "pause_handling",
    "backchannel": "backchannel",
    "backchanneling": "backchannel",
    "turn_taking": "turn_taking",
    "smooth_turn_taking": "turn_taking",
    "interruption": "interruption",
    "user_interruption": "interruption",
}

# Recommended text prompts per Full-Duplex-Bench category (from PersonaPlex README)
CATEGORY_PROMPTS = {
    "pause_handling": "You enjoy having a good conversation.",
    "backchannel": "You enjoy having a good conversation.",
    "turn_taking": "You enjoy having a good conversation.",
    "interruption": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
}


def _load_metadata(dataset_dir: str, category: str) -> list[dict]:
    """Load metadata.json for a category, or synthesize from audio files."""
    cat_dir = Path(dataset_dir) / category
    meta_path = cat_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)

    # Fallback: scan audio directory
    audio_dir = cat_dir / "audio"
    if not audio_dir.exists():
        audio_dir = cat_dir  # audio files directly in category dir
    wav_files = sorted(audio_dir.glob("*.wav"))
    return [{"audio": str(f), "id": f.stem} for f in wav_files]


def run_evaluation(
    dataset_dir: str,
    output_dir: str,
    categories: List[str],
    pipeline: VoiceToVoicePipeline,
    threshold_db: float = -40.0,
) -> FullDuplexBenchResult:
    """Run Full-Duplex-Bench evaluation on specified categories."""
    os.makedirs(output_dir, exist_ok=True)
    bench_result = FullDuplexBenchResult()

    for cat_name in categories:
        canonical = CATEGORY_MAP.get(cat_name, cat_name)
        print(f"\n=== Evaluating: {canonical} ===")

        metadata = _load_metadata(dataset_dir, canonical)
        if not metadata:
            print(f"  No samples found for {canonical}, skipping.")
            continue

        cat_output_dir = os.path.join(output_dir, canonical)
        os.makedirs(cat_output_dir, exist_ok=True)

        # Update text prompt for this category
        if canonical in CATEGORY_PROMPTS:
            pipeline.text_prompt = CATEGORY_PROMPTS[canonical]

        for i, sample in enumerate(metadata):
            audio_path = sample.get("audio", "")
            sample_id = sample.get("id", f"sample_{i:04d}")
            print(f"  [{i+1}/{len(metadata)}] Processing {sample_id}...")

            if not os.path.exists(audio_path):
                # Try relative to dataset dir
                alt_path = os.path.join(dataset_dir, canonical, audio_path)
                if os.path.exists(alt_path):
                    audio_path = alt_path
                else:
                    print(f"    Audio not found: {audio_path}, skipping.")
                    continue

            out_wav = os.path.join(cat_output_dir, f"{sample_id}_output.wav")
            result = pipeline.run(input_wav=audio_path, output_wav=out_wav)

            if result.audio_pcm is None:
                print(f"    No output generated for {sample_id}, skipping.")
                continue

            # Save text output
            text_path = os.path.join(cat_output_dir, f"{sample_id}_text.json")
            with open(text_path, "w") as f:
                json.dump({"text": result.text, "tokens": result.tokens}, f, ensure_ascii=False, indent=2)

            # Compute category-specific metrics
            model_audio = result.audio_pcm
            sr = result.sample_rate

            if canonical == "pause_handling":
                pause_start = sample.get("pause_start", 2.0)
                pause_end = sample.get("pause_end", 5.0)
                ph = evaluate_pause_handling(model_audio, sr, pause_start, pause_end, threshold_db)
                bench_result.pause_handling.append(ph)
                print(f"    Pause: spoke_during={ph.model_spoke_during_pause}")

            elif canonical == "turn_taking":
                user_end = sample.get("user_end_time", 3.0)
                tt = evaluate_turn_taking(model_audio, sr, user_end, threshold_db)
                bench_result.turn_taking.append(tt)
                print(f"    Turn-taking gap: {tt.gap_duration*1000:.0f}ms")

            elif canonical == "interruption":
                int_time = sample.get("interruption_time", 3.0)
                ir = evaluate_interruption(model_audio, sr, int_time, threshold_db)
                bench_result.interruption.append(ir)
                print(f"    Interruption reaction: {ir.reaction_time*1000:.0f}ms")

    # Write summary
    summary = bench_result.summary()
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Summary written to {summary_path} ===")
    for cat, metrics in summary.items():
        print(f"  {cat}: {metrics}")

    return bench_result


def main():
    parser = argparse.ArgumentParser(
        description="Full-Duplex-Bench evaluation runner for PersonaPlex/Moshi."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=str,
        help="Path to the Full-Duplex-Bench dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Path to write evaluation results.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["pause_handling", "turn_taking", "interruption"],
        choices=list(CATEGORY_MAP.keys()),
        help="Which evaluation categories to run (default: pause_handling turn_taking interruption).",
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=-40.0,
        help="VAD energy threshold in dB (default: -40).",
    )

    # Voice / model args
    parser.add_argument("--voice-prompt", default="NATF2.pt", type=str)
    parser.add_argument("--voice-prompt-dir", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temp-audio", type=float, default=0.8)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--topk-audio", type=int, default=250)
    parser.add_argument("--topk-text", type=int, default=25)

    args = parser.parse_args()

    print("Initializing VoiceToVoice pipeline for evaluation...")
    pipeline = VoiceToVoicePipeline(
        hf_repo=args.hf_repo,
        device=args.device,
        mimi_weight=args.mimi_weight,
        moshi_weight=args.moshi_weight,
        tokenizer_path=args.tokenizer,
        voice_prompt=args.voice_prompt,
        voice_prompt_dir=args.voice_prompt_dir,
        text_prompt="",  # will be set per category
        seed=args.seed,
        temp_audio=args.temp_audio,
        temp_text=args.temp_text,
        topk_audio=args.topk_audio,
        topk_text=args.topk_text,
        greedy=args.greedy,
        cpu_offload=args.cpu_offload,
    )

    run_evaluation(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        categories=args.categories,
        pipeline=pipeline,
        threshold_db=args.threshold_db,
    )


if __name__ == "__main__":
    with torch.no_grad():
        main()
