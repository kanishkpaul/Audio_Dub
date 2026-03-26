"""
Audio Duration Adjustment using Pause-Aware TD-PSOLA
=====================================================
Intelligently adjusts audio duration while preserving speaker identity and quality.

Uses Praat's TD-PSOLA (Time-Domain Pitch-Synchronous Overlap-Add) algorithm for 
pitch-preserving speed changes, combined with smart pause detection and manipulation.

Methods:
- Shrinking: trim pauses → remove all pauses → speed up (max 1.2x)
- Stretching: slow down (0.909x) → inject silence in pauses
"""

import numpy as np
import soundfile as sf
import librosa
import parselmouth

# ─────────────────────────────────────────────────────────────────────────────
# PITCH-PRESERVED SPEED CHANGE VIA PRAAT TD-PSOLA
# ─────────────────────────────────────────────────────────────────────────────


def change_speed(wav: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """
    Time-stretch audio by `speed` without altering pitch.
    
    Args:
        wav: Audio array (mono float32)
        sr: Sample rate (Hz)
        speed: Speed multiplier
            - speed > 1.0 → faster (shorter duration)
            - speed < 1.0 → slower (longer duration)
    
    Returns:
        Speed-adjusted audio (same sample rate as input)
    
    Algorithm: Praat TD-PSOLA
        - Preserves speaker identity and pitch characteristics
        - Best quality for speech audio
        - More natural-sounding than librosa phase vocoder
    """
    sound = parselmouth.Sound(wav.astype(np.float32), sampling_frequency=sr)
    manipulation = parselmouth.praat.call(sound, "To Manipulation", 0.01, 75, 600)

    dur_tier = parselmouth.praat.call("Create DurationTier", "dur", 0.0, sound.duration)
    parselmouth.praat.call(dur_tier, "Add point", 0.0, 1.0 / speed)
    parselmouth.praat.call(dur_tier, "Add point", sound.duration, 1.0 / speed)
    parselmouth.praat.call([manipulation, dur_tier], "Replace duration tier")

    resynth = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
    out = resynth.values[0].astype(wav.dtype)
    out_sr = int(resynth.sampling_frequency)

    if out_sr != sr:
        out = librosa.resample(out.astype(np.float32), orig_sr=out_sr, target_sr=sr).astype(wav.dtype)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# PAUSE DETECTION (VAD)
# ─────────────────────────────────────────────────────────────────────────────


def detect_pauses(
    wav: np.ndarray,
    sr: int,
    frame_ms: int = 20,
    energy_threshold: float = 0.01,
    min_pause_ms: int = 80,
) -> list[tuple[float, float]]:
    """
    Detect silent/pause regions using energy-based VAD.
    
    Args:
        wav: Audio array (mono float32)
        sr: Sample rate (Hz)
        frame_ms: Frame length for RMS computation (default 20ms)
        energy_threshold: Silence threshold as fraction of max RMS
                         (default 0.01 = 1% of max energy)
        min_pause_ms: Minimum pause duration to detect (default 80ms)
    
    Returns:
        List of (start_sec, end_sec) tuples for each detected pause
    """
    frame_len = int(sr * frame_ms / 1000)
    frames = [wav[i : i + frame_len] for i in range(0, len(wav) - frame_len, frame_len)]
    rms_vals = np.array([np.sqrt(np.mean(f**2)) for f in frames])
    threshold = rms_vals.max() * energy_threshold
    is_silence = rms_vals < threshold

    pauses, in_pause, pause_start = [], False, 0.0
    for i, silent in enumerate(is_silence):
        t = i * frame_ms / 1000.0
        if silent and not in_pause:
            in_pause, pause_start = True, t
        elif not silent and in_pause:
            in_pause = False
            if (t - pause_start) * 1000 >= min_pause_ms:
                pauses.append((pause_start, t))
    if in_pause:
        t = len(frames) * frame_ms / 1000.0
        if (t - pause_start) * 1000 >= min_pause_ms:
            pauses.append((pause_start, t))

    return pauses


# ─────────────────────────────────────────────────────────────────────────────
# SHRINKING (Making audio shorter)
# ─────────────────────────────────────────────────────────────────────────────


def shrink_audio(
    wav: np.ndarray,
    sr: int,
    pauses: list[tuple[float, float]],
    target_duration: float,
) -> np.ndarray:
    """
    Reduce audio length to target_duration using a smart priority chain:

    Stage 1: Evenly trim all pauses
        If total pause duration >= required reduction, trim each pause equally.
        Speech segments remain untouched.

    Stage 2: Remove ALL pauses, then speed up
        If pauses aren't sufficient, remove them entirely.
        Then calculate required speed-up: speed = current_duration / target_duration
        Applied via Praat TD-PSOLA (max 1.2x to preserve quality).

    Stage 3: Hard limit warning
        If even 1.2x speed isn't enough, emit warning and apply 1.2x anyway.

    Args:
        wav: Audio array (mono float32)
        sr: Sample rate (Hz)
        pauses: List of (start_sec, end_sec) tuples from detect_pauses()
        target_duration: Desired output duration in seconds

    Returns:
        Shortened audio array
    """
    original_duration = len(wav) / sr
    need_to_remove = original_duration - target_duration

    print(
        f"[SHRINK] {original_duration:.2f}s → {target_duration:.2f}s "
        f"(need to remove {need_to_remove:.2f}s)"
    )

    total_pause_dur = sum(e - s for s, e in pauses) if pauses else 0.0

    # ── Stage 1: trim pauses evenly ───────────────────────────────────────────
    if pauses and need_to_remove <= total_pause_dur:
        cut_per_pause = need_to_remove / len(pauses)
        print(
            f"[SHRINK] Stage 1 – cutting {cut_per_pause * 1000:.0f}ms "
            f"from each of {len(pauses)} pause(s)"
        )

        segments, prev = [], 0
        for pause_start, pause_end in pauses:
            s_samp = int(pause_start * sr)
            e_samp = int(pause_end * sr)
            keep = max(0.0, (pause_end - pause_start) - cut_per_pause)
            segments.append(wav[prev:s_samp])
            segments.append(wav[s_samp : s_samp + int(keep * sr)])
            prev = e_samp
        segments.append(wav[prev:])
        out = np.concatenate(segments)

    # ── Stage 2 +3: remove all pauses, then speed up via Praat ────────────────
    else:
        print(f"[SHRINK] Stage 2 – removing all pauses " f"({total_pause_dur:.2f}s total)")
        segs, prev = [], 0
        for pause_start, pause_end in pauses:
            segs.append(wav[prev : int(pause_start * sr)])
            prev = int(pause_end * sr)
        segs.append(wav[prev:])
        no_pause_wav = np.concatenate(segs) if segs else wav.copy()
        duration_no_pause = len(no_pause_wav) / sr

        if duration_no_pause <= target_duration:
            print(
                f"[SHRINK] After pause removal: {duration_no_pause:.2f}s "
                f"<= target {target_duration:.2f}s — done."
            )
            out = no_pause_wav
        else:
            speed_needed = duration_no_pause / target_duration
            if speed_needed <= 1.2:
                print(
                    f"[SHRINK] Stage 2 – applying {speed_needed:.3f}x "
                    f"speed-up via Praat TD-PSOLA"
                )
                out = change_speed(no_pause_wav, sr, speed_needed)
            else:
                achieved = duration_no_pause / 1.2
                print(
                    f"[SHRINK] WARNING: target requires {speed_needed:.3f}x "
                    f"speed-up which exceeds the 1.2x limit!"
                )
                print(
                    f"[SHRINK] Applying maximum 1.2x — "
                    f"achieved ~{achieved:.2f}s (target: {target_duration:.2f}s)"
                )
                out = change_speed(no_pause_wav, sr, 1.2)

    print(f"[RESULT] Final duration: {len(out) / sr:.2f}s")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STRETCHING (Making audio longer)
# ─────────────────────────────────────────────────────────────────────────────


def stretch_audio(
    wav: np.ndarray,
    sr: int,
    pauses: list[tuple[float, float]],
    target_duration: float,
) -> np.ndarray:
    """
    Increase audio length to target_duration using an additive approach:

    Step 1: Slow down entire audio to 0.909x (1/1.1) via Praat TD-PSOLA
        This naturally lengthens speech while preserving speaker identity.
        Example: 25s → 25 × 1.1 = 27.5s
        Pause timestamps are rescaled by ×1.1 to stay aligned.

    Step 2: If still short, insert silence into pauses
        Remaining gap is distributed evenly across all detected pauses.
        If no pauses exist, silence is appended at the end.
        No upper limit on expansion.

    Args:
        wav: Audio array (mono float32)
        sr: Sample rate (Hz)
        pauses: List of (start_sec, end_sec) tuples from detect_pauses()
        target_duration: Desired output duration in seconds

    Returns:
        Stretched audio array
    """
    original_duration = len(wav) / sr
    print(f"\n[STRETCH] {original_duration:.2f}s → {target_duration:.2f}s")

    # Step 1 – slow down to 1/1.1x (pitch preserved)
    slow_speed = 1.0 / 1.1  # ≈ 0.9091
    print(f"[STRETCH] Step 1 – slowing down to {slow_speed:.4f}x via Praat TD-PSOLA")
    slowed = change_speed(wav, sr, slow_speed)
    slowed_duration = len(slowed) / sr
    print(f"[STRETCH] After slowdown: {slowed_duration:.2f}s " f"(was {original_duration:.2f}s)")

    # Scale pause timestamps to the new (longer) timeline
    scaled_pauses = [(s * 1.1, e * 1.1) for s, e in pauses]
    extra_needed = target_duration - slowed_duration

    if extra_needed <= 0:
        # Slowing down already met or exceeded the target
        print("[STRETCH] Slowdown alone meets or exceeds target — " "no pause expansion needed.")
        out = slowed
    elif not scaled_pauses:
        # No pauses detected — append silence at the end
        print(f"[STRETCH] No pauses found; appending " f"{extra_needed:.2f}s silence at end.")
        out = np.concatenate(
            [
                slowed,
                np.zeros(int(extra_needed * sr), dtype=slowed.dtype),
            ]
        )
    else:
        # Step 2 – distribute remaining gap evenly across pauses
        extra_per_pause = extra_needed / len(scaled_pauses)
        print(
            f"[STRETCH] Step 2 – adding {extra_per_pause * 1000:.0f}ms "
            f"to each of {len(scaled_pauses)} pause(s)"
        )

        silence_chunk = np.zeros(int(extra_per_pause * sr), dtype=slowed.dtype)
        segments, prev = [], 0

        for pause_start, pause_end in scaled_pauses:
            end_sample = int(pause_end * sr)
            segments.append(slowed[prev:end_sample])
            segments.append(silence_chunk)
            prev = end_sample
        segments.append(slowed[prev:])

        out = np.concatenate(segments)

    print(f"[RESULT] Final duration: {len(out) / sr:.2f}s")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER: Smart decision on whether to shrink or stretch
# ─────────────────────────────────────────────────────────────────────────────


def adjust_audio_duration(
    wav: np.ndarray,
    sr: int,
    target_duration: float,
    energy_threshold: float = 0.01,
    min_pause_ms: int = 80,
) -> np.ndarray:
    """
    Intelligently adjust audio duration to match target.

    Decides whether to shrink or stretch based on current vs target duration.
    Skips processing if already within 50ms of target (to avoid artifacts).

    Args:
        wav: Audio array (mono float32)
        sr: Sample rate (Hz)
        target_duration: Desired output duration in seconds
        energy_threshold: VAD energy threshold (0.01 = 1% of max)
        min_pause_ms: Minimum pause length to detect (80ms)

    Returns:
        Duration-adjusted audio array
    """
    original_duration = len(wav) / sr
    delta = target_duration - original_duration

    # Skip if already within tolerance
    if abs(delta) < 0.05:
        print(
            f"[ADJUST] Already within 50ms of target "
            f"({original_duration:.2f}s ≈ {target_duration:.2f}s) — skipping."
        )
        return wav

    # Detect pauses
    pauses = detect_pauses(wav, sr, energy_threshold=energy_threshold, min_pause_ms=min_pause_ms)
    if pauses:
        total_pause_time = sum(e - s for s, e in pauses)
        print(f"[ADJUST] Found {len(pauses)} pause(s), total {total_pause_time:.2f}s")
    else:
        print("[ADJUST] No pauses detected")

    # Route to shrink or stretch
    if delta < 0:
        return shrink_audio(wav, sr, pauses, target_duration)
    else:
        return stretch_audio(wav, sr, pauses, target_duration)
