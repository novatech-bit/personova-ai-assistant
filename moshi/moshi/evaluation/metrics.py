# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Metric computation utilities for Full-Duplex-Bench evaluation.

Implements the four evaluation categories from the Full-Duplex-Bench paper
(arXiv:2503.04721):

  1. Pause Handling       - Measures the model's ability to remain silent
                            during user pauses of varying length.
  2. Backchanneling       - Detects appropriate listener feedback signals.
  3. Smooth Turn-Taking   - Evaluates turn transition latency and naturalness.
  4. User Interruption    - Measures how quickly the model stops speaking
                            after the user interrupts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Energy / VAD helpers
# ---------------------------------------------------------------------------

def compute_rms_energy(pcm: np.ndarray, frame_size: int = 1920, hop_size: int = 960) -> np.ndarray:
    """Compute frame-level RMS energy for a mono PCM signal."""
    n_frames = max(1, (len(pcm) - frame_size) // hop_size + 1)
    energies = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_size
        frame = pcm[start: start + frame_size]
        energies[i] = np.sqrt(np.mean(frame ** 2))
    return energies


def simple_vad(pcm: np.ndarray, sample_rate: int, threshold_db: float = -40.0,
               frame_duration_ms: float = 80.0) -> List[tuple[float, float]]:
    """Simple energy-based VAD returning speech segments as (start_sec, end_sec) pairs."""
    frame_size = int(sample_rate * frame_duration_ms / 1000.0)
    hop_size = frame_size // 2
    energies = compute_rms_energy(pcm, frame_size, hop_size)

    threshold = 10 ** (threshold_db / 20.0)
    is_speech = energies > threshold

    segments: List[tuple[float, float]] = []
    in_segment = False
    seg_start = 0.0
    for i, active in enumerate(is_speech):
        t = i * hop_size / sample_rate
        if active and not in_segment:
            seg_start = t
            in_segment = True
        elif not active and in_segment:
            segments.append((seg_start, t))
            in_segment = False
    if in_segment:
        segments.append((seg_start, len(pcm) / sample_rate))
    return segments


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PauseHandlingResult:
    """Pause handling evaluation for a single sample."""
    pause_duration: float
    model_spoke_during_pause: bool
    false_start_time: Optional[float] = None


@dataclass
class BackchannelResult:
    """Backchannel evaluation for a single sample."""
    expected_backchannel_windows: List[tuple[float, float]] = field(default_factory=list)
    detected_backchannels: List[tuple[float, float]] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0


@dataclass
class TurnTakingResult:
    """Turn-taking evaluation for a single sample."""
    user_end_time: float = 0.0
    model_start_time: float = 0.0
    gap_duration: float = 0.0
    overlap_duration: float = 0.0


@dataclass
class InterruptionResult:
    """User interruption evaluation for a single sample."""
    interruption_time: float = 0.0
    model_stop_time: float = 0.0
    reaction_time: float = 0.0


@dataclass
class FullDuplexBenchResult:
    """Aggregated evaluation result across all categories."""
    pause_handling: List[PauseHandlingResult] = field(default_factory=list)
    backchanneling: List[BackchannelResult] = field(default_factory=list)
    turn_taking: List[TurnTakingResult] = field(default_factory=list)
    interruption: List[InterruptionResult] = field(default_factory=list)

    def summary(self) -> dict:
        """Compute aggregated metrics across all samples."""
        result = {}

        if self.pause_handling:
            n = len(self.pause_handling)
            false_starts = sum(1 for p in self.pause_handling if p.model_spoke_during_pause)
            result["pause_handling"] = {
                "n_samples": n,
                "false_start_rate": false_starts / n if n > 0 else 0.0,
                "patience_rate": 1.0 - (false_starts / n) if n > 0 else 1.0,
            }

        if self.backchanneling:
            n = len(self.backchanneling)
            avg_prec = np.mean([b.precision for b in self.backchanneling])
            avg_rec = np.mean([b.recall for b in self.backchanneling])
            f1 = (2 * avg_prec * avg_rec / (avg_prec + avg_rec)) if (avg_prec + avg_rec) > 0 else 0.0
            result["backchanneling"] = {
                "n_samples": n,
                "avg_precision": float(avg_prec),
                "avg_recall": float(avg_rec),
                "f1": float(f1),
            }

        if self.turn_taking:
            n = len(self.turn_taking)
            gaps = [t.gap_duration for t in self.turn_taking if t.gap_duration > 0]
            overlaps = [t.overlap_duration for t in self.turn_taking if t.overlap_duration > 0]
            result["turn_taking"] = {
                "n_samples": n,
                "avg_gap_ms": float(np.mean(gaps) * 1000) if gaps else 0.0,
                "avg_overlap_ms": float(np.mean(overlaps) * 1000) if overlaps else 0.0,
                "median_gap_ms": float(np.median(gaps) * 1000) if gaps else 0.0,
            }

        if self.interruption:
            n = len(self.interruption)
            reaction_times = [i.reaction_time for i in self.interruption]
            result["interruption"] = {
                "n_samples": n,
                "avg_reaction_time_ms": float(np.mean(reaction_times) * 1000),
                "median_reaction_time_ms": float(np.median(reaction_times) * 1000),
                "p90_reaction_time_ms": float(np.percentile(reaction_times, 90) * 1000),
            }

        return result


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

def evaluate_pause_handling(
    model_audio: np.ndarray,
    sample_rate: int,
    pause_start: float,
    pause_end: float,
    threshold_db: float = -40.0,
) -> PauseHandlingResult:
    """Check whether the model spoke during a known pause in the user's speech."""
    pause_samples_start = int(pause_start * sample_rate)
    pause_samples_end = int(pause_end * sample_rate)
    pause_audio = model_audio[pause_samples_start:pause_samples_end]

    segments = simple_vad(pause_audio, sample_rate, threshold_db)
    spoke = len(segments) > 0
    false_start = segments[0][0] + pause_start if spoke else None

    return PauseHandlingResult(
        pause_duration=pause_end - pause_start,
        model_spoke_during_pause=spoke,
        false_start_time=false_start,
    )


def evaluate_turn_taking(
    model_audio: np.ndarray,
    sample_rate: int,
    user_end_time: float,
    threshold_db: float = -40.0,
) -> TurnTakingResult:
    """Measure the gap/overlap between user turn end and model response start."""
    after_user = model_audio[int(user_end_time * sample_rate):]
    segments = simple_vad(after_user, sample_rate, threshold_db)

    if segments:
        model_start_offset = segments[0][0]
        model_start_time = user_end_time + model_start_offset
        gap = model_start_offset
        return TurnTakingResult(
            user_end_time=user_end_time,
            model_start_time=model_start_time,
            gap_duration=max(0, gap),
            overlap_duration=max(0, -gap),
        )
    return TurnTakingResult(user_end_time=user_end_time)


def evaluate_interruption(
    model_audio: np.ndarray,
    sample_rate: int,
    interruption_time: float,
    threshold_db: float = -40.0,
    max_window: float = 3.0,
) -> InterruptionResult:
    """Measure how quickly the model stops speaking after user interruption."""
    int_sample = int(interruption_time * sample_rate)
    end_sample = min(len(model_audio), int(int_sample + max_window * sample_rate))
    window = model_audio[int_sample:end_sample]

    segments = simple_vad(window, sample_rate, threshold_db)
    if segments:
        last_end = segments[-1][1]
        model_stop_time = interruption_time + last_end
        reaction = last_end
    else:
        model_stop_time = interruption_time
        reaction = 0.0

    return InterruptionResult(
        interruption_time=interruption_time,
        model_stop_time=model_stop_time,
        reaction_time=reaction,
    )
