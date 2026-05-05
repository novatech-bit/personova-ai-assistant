# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CLI entrypoint for text-to-text inference using PersonaPlex/Moshi.

Usage:
    python -m moshi.text_to_text --user-text "What is the speed of light?"

The model processes the user text by feeding silence as audio input and
captures only the generated text response.
"""

import argparse
import json

import torch

from .models import loaders


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-text inference using PersonaPlex/Moshi."
    )
    parser.add_argument(
        "--user-text",
        required=True,
        type=str,
        help="The user's text input to generate a response for.",
    )
    parser.add_argument(
        "--text-prompt",
        default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
        type=str,
        help="System text prompt defining the persona.",
    )
    parser.add_argument(
        "--voice-prompt",
        default="NATF2.pt",
        type=str,
        help="Voice prompt filename (e.g. 'NATF2.pt').",
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help="Directory containing voice prompt files. If omitted, downloaded from HF.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of text tokens to generate (default: 200).",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=10.0,
        help="Duration of silence audio to feed in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--output-text",
        type=str,
        help="Optional path to write output text/tokens as JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42424242,
        help="Random seed (default: 42424242). Use -1 to disable.",
    )

    # Model assets
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local Moshi checkpoint.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local Mimi checkpoint.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into (defaults to PersonaPlex).",
    )

    # Sampling
    parser.add_argument("--temp-text", type=float, default=0.7, help="Text sampling temperature.")
    parser.add_argument("--temp-audio", type=float, default=0.8, help="Audio sampling temperature.")
    parser.add_argument("--topk-text", type=int, default=25, help="Text top-k sampling.")
    parser.add_argument("--topk-audio", type=int, default=250, help="Audio top-k sampling.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding.")

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (default: 'cuda')."
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload LM layers to CPU when GPU memory is insufficient.",
    )

    args = parser.parse_args()

    from .pipeline import TextToTextPipeline

    print(f"Initializing text-to-text pipeline on {args.device}...")
    pipeline = TextToTextPipeline(
        hf_repo=args.hf_repo,
        device=args.device,
        mimi_weight=args.mimi_weight,
        moshi_weight=args.moshi_weight,
        tokenizer_path=args.tokenizer,
        voice_prompt=args.voice_prompt,
        voice_prompt_dir=args.voice_prompt_dir,
        text_prompt=args.text_prompt,
        seed=args.seed,
        temp_audio=args.temp_audio,
        temp_text=args.temp_text,
        topk_audio=args.topk_audio,
        topk_text=args.topk_text,
        greedy=args.greedy,
        cpu_offload=args.cpu_offload,
    )

    print(f"User text: {args.user_text}")
    print("Generating response...")
    result = pipeline.run(
        user_text=args.user_text,
        max_tokens=args.max_tokens,
        silence_duration=args.silence_duration,
    )

    print(f"\nResponse: {result.text}")

    if args.output_text:
        with open(args.output_text, "w") as f:
            json.dump({"text": result.text, "tokens": result.tokens}, f, ensure_ascii=False, indent=2)
        print(f"Wrote output to {args.output_text}")


with torch.no_grad():
    main()
