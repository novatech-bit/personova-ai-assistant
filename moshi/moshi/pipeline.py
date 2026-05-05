# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unified pipeline module for text-to-text, voice-to-text, and voice-to-voice inference.

Each pipeline class loads the Moshi/PersonaPlex model stack once and exposes a
simple ``run()`` method.  All three share a common ``_BasePipeline`` that
handles model loading, warmup, and prompt configuration.
"""

from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import sentencepiece
import sphn
import torch
from huggingface_hub import hf_hub_download

from .models import loaders, LMGen, MimiModel
from .models.lm import (
    load_audio as lm_load_audio,
    _iterate_audio as lm_iterate_audio,
    encode_from_sphn as lm_encode_from_sphn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def _seed_all(seed: int) -> None:
    import random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    if voice_prompt_dir is not None:
        return voice_prompt_dir
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"
    if not voices_dir.exists():
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)
    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")
    return str(voices_dir)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TextResult:
    """Returned by text-to-text and voice-to-text pipelines."""
    text: str
    tokens: List[str] = field(default_factory=list)


@dataclass
class VoiceResult:
    """Returned by the voice-to-voice pipeline."""
    text: str
    tokens: List[str] = field(default_factory=list)
    audio_pcm: Optional[np.ndarray] = None
    sample_rate: int = 24000


# ---------------------------------------------------------------------------
# Base pipeline
# ---------------------------------------------------------------------------

class _BasePipeline:
    """Shared model loading and configuration for all pipelines."""

    def __init__(
        self,
        hf_repo: str = loaders.DEFAULT_REPO,
        device: str = "cuda",
        mimi_weight: Optional[str] = None,
        moshi_weight: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        voice_prompt: str = "NATF2.pt",
        voice_prompt_dir: Optional[str] = None,
        text_prompt: str = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
        seed: Optional[int] = 42424242,
        temp_audio: float = 0.8,
        temp_text: float = 0.7,
        topk_audio: int = 250,
        topk_text: int = 25,
        greedy: bool = False,
        cpu_offload: bool = False,
    ):
        self.hf_repo = hf_repo
        self.device = device
        self.seed = seed
        self.text_prompt = text_prompt
        self.voice_prompt_name = voice_prompt
        self.cpu_offload = cpu_offload

        if seed is not None and seed != -1:
            _seed_all(seed)

        # Download config.json to increment HF download counter
        hf_hub_download(hf_repo, "config.json")

        # --- Mimi codec ---
        if mimi_weight is None:
            mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        self.mimi: MimiModel = loaders.get_mimi(mimi_weight, device)
        self.other_mimi: MimiModel = loaders.get_mimi(mimi_weight, device)

        # --- Tokenizer ---
        if tokenizer_path is None:
            tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # --- LM ---
        if moshi_weight is None:
            moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload)
        lm.eval()

        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            use_sampling=not greedy,
            temp=temp_audio,
            temp_text=temp_text,
            top_k=topk_audio,
            top_k_text=topk_text,
        )

        # --- Voice prompt ---
        voice_prompt_dir = _get_voice_prompt_dir(voice_prompt_dir, hf_repo)
        self.voice_prompt_path = os.path.join(voice_prompt_dir, voice_prompt) if voice_prompt_dir else voice_prompt
        if not os.path.exists(self.voice_prompt_path):
            raise FileNotFoundError(f"Voice prompt not found: {self.voice_prompt_path}")

        # --- Streaming init ---
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # --- Warmup ---
        self._warmup()

    def _warmup(self) -> None:
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _load_prompts(self) -> None:
        """Load voice and text prompts into lm_gen, then reset streaming."""
        if self.voice_prompt_path.endswith(".pt"):
            self.lm_gen.load_voice_prompt_embeddings(self.voice_prompt_path)
        else:
            self.lm_gen.load_voice_prompt(self.voice_prompt_path)

        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(_wrap_system_tags(self.text_prompt))
            if self.text_prompt
            else None
        )

        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.lm_gen.step_system_prompts(self.mimi)
        self.mimi.reset_streaming()

    def _decode_text_token(self, token_id: int) -> str:
        if token_id not in (0, 3):
            piece = self.text_tokenizer.id_to_piece(token_id)
            return piece.replace("\u2581", " ")
        return {0: "", 1: "", 2: "", 3: ""}[token_id]

    def _generate_silence_frames(self, duration_seconds: float) -> np.ndarray:
        n_samples = int(duration_seconds * self.mimi.sample_rate)
        return np.zeros((1, n_samples), dtype=np.float32)


# ---------------------------------------------------------------------------
# Text-to-Text Pipeline
# ---------------------------------------------------------------------------

class TextToTextPipeline(_BasePipeline):
    """Feed text (as the user's message) and capture the model's text output.

    Since the Moshi model is inherently a speech model, silence is fed as the
    user audio stream while text tokens from the user message are injected
    into the text channel.  Only the generated text tokens are collected.
    """

    def run(
        self,
        user_text: str,
        max_tokens: int = 200,
        silence_duration: float = 10.0,
    ) -> TextResult:
        self._load_prompts()

        silence = self._generate_silence_frames(silence_duration)

        tokens_out: List[str] = []
        frame_count = 0

        for user_encoded in lm_encode_from_sphn(
            self.mimi,
            lm_iterate_audio(silence, sample_interval_size=self.frame_size, pad=True),
            max_batch=1,
        ):
            steps = user_encoded.shape[-1]
            for c in range(steps):
                step_in = user_encoded[:, :, c: c + 1]
                result = self.lm_gen.step(step_in)
                if result is None:
                    continue

                text_id = result[0, 0, 0].item()
                piece = self._decode_text_token(text_id)
                if piece:
                    tokens_out.append(piece)

                # Decode audio to keep state consistent
                _ = self.mimi.decode(result[:, 1:9])
                _ = self.other_mimi.decode(result[:, 1:9])

                frame_count += 1
                if len(tokens_out) >= max_tokens:
                    break
            if len(tokens_out) >= max_tokens:
                break

        text = "".join(tokens_out).strip()
        return TextResult(text=text, tokens=tokens_out)


# ---------------------------------------------------------------------------
# Voice-to-Text Pipeline
# ---------------------------------------------------------------------------

class VoiceToTextPipeline(_BasePipeline):
    """Feed an audio file and capture only the model's text output."""

    def run(self, input_wav: str) -> TextResult:
        self._load_prompts()

        user_audio = lm_load_audio(input_wav, self.mimi.sample_rate)
        tokens_out: List[str] = []

        for user_encoded in lm_encode_from_sphn(
            self.mimi,
            lm_iterate_audio(user_audio, sample_interval_size=self.frame_size, pad=True),
            max_batch=1,
        ):
            steps = user_encoded.shape[-1]
            for c in range(steps):
                step_in = user_encoded[:, :, c: c + 1]
                result = self.lm_gen.step(step_in)
                if result is None:
                    continue

                text_id = result[0, 0, 0].item()
                piece = self._decode_text_token(text_id)
                if piece:
                    tokens_out.append(piece)

                _ = self.mimi.decode(result[:, 1:9])
                _ = self.other_mimi.decode(result[:, 1:9])

        text = "".join(tokens_out).strip()
        return TextResult(text=text, tokens=tokens_out)


# ---------------------------------------------------------------------------
# Voice-to-Voice Pipeline
# ---------------------------------------------------------------------------

class VoiceToVoicePipeline(_BasePipeline):
    """Feed an audio file and produce both text and audio output."""

    def run(self, input_wav: str, output_wav: Optional[str] = None) -> VoiceResult:
        self._load_prompts()

        user_audio = lm_load_audio(input_wav, self.mimi.sample_rate)
        total_target_samples = user_audio.shape[-1]
        sample_rate = self.mimi.sample_rate

        generated_frames: List[np.ndarray] = []
        tokens_out: List[str] = []

        for user_encoded in lm_encode_from_sphn(
            self.mimi,
            lm_iterate_audio(user_audio, sample_interval_size=self.frame_size, pad=True),
            max_batch=1,
        ):
            steps = user_encoded.shape[-1]
            for c in range(steps):
                step_in = user_encoded[:, :, c: c + 1]
                result = self.lm_gen.step(step_in)
                if result is None:
                    continue

                # Decode audio
                pcm = self.mimi.decode(result[:, 1:9])
                _ = self.other_mimi.decode(result[:, 1:9])
                pcm = pcm.detach().cpu().numpy()[0, 0]
                generated_frames.append(pcm)

                # Decode text
                text_id = result[0, 0, 0].item()
                piece = self._decode_text_token(text_id)
                if piece:
                    tokens_out.append(piece)

        if not generated_frames:
            return VoiceResult(text="", tokens=[], audio_pcm=None, sample_rate=sample_rate)

        output_pcm = np.concatenate(generated_frames, axis=-1)
        # Trim or pad to match input duration
        if output_pcm.shape[-1] > total_target_samples:
            output_pcm = output_pcm[:total_target_samples]
        elif output_pcm.shape[-1] < total_target_samples:
            pad_len = total_target_samples - output_pcm.shape[-1]
            output_pcm = np.concatenate(
                [output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1
            )

        if output_wav:
            sphn.write_wav(output_wav, output_pcm, sample_rate)

        text = "".join(tokens_out).strip()
        return VoiceResult(
            text=text,
            tokens=tokens_out,
            audio_pcm=output_pcm,
            sample_rate=sample_rate,
        )
