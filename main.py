import argparse
import json
import os
import subprocess
import tempfile
import uuid
import time
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa

from audio_adjustment import adjust_audio_duration
from Vocal_Music_Separation import vocal_music_separator
from Speech_Overlap import detect_overlaps
from Speaker_Diarization import perform_diarization_and_extract
from Speaker_Separation import separate_speakers
from Speaker_Identification import match_and_merge_speaker
from Reference_Extraction import get_tts_reference
from ASR import transcribe_audio, WHISPER_ID
from Qwen3llm import translate_fragment
from Qwen3tts import load_tts_model, generate_voice_clone
from helper import ensure_wav, load_mono, save_wav, load_json, save_json, load_env_value

DEFAULT_TEMP_DIR = "temp"
DEFAULT_SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_SEPARATION_MODEL = "lichenda/wsj0_2mix_skim_noncausal"
DEFAULT_MATCH_THRESHOLD = 0.60
DEFAULT_SEPARATION_DEVICE = "cuda"
DEFAULT_SR = 16000
DEFAULT_CACHE_DIR = "cache"


def time_stretch_to_duration(audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    """
    Adjust audio duration to match target using pause-aware, pitch-preserving method.
    
    Uses Praat TD-PSOLA for speed changes to maintain speaker identity.
    Intelligently handles shrinking (trim pauses → speed up) and stretching
    (slow down → inject silence in pauses).
    
    Args:
        audio: Audio array (mono float32)
        sr: Sample rate (Hz)
        target_duration: Desired output duration in seconds
    
    Returns:
        Duration-adjusted audio array
    """
    if len(audio) == 0 or target_duration <= 0:
        return audio
    
    current_duration = len(audio) / sr
    if current_duration <= 0:
        return audio
    
    try:
        return adjust_audio_duration(audio, sr, target_duration)
    except Exception as e:
        print(f"[ERROR] Audio adjustment failed: {e}")
        print(f"[WARN] Returning original audio unchanged.")
        return audio


def overlay_audio(base: np.ndarray, overlay: np.ndarray, start_sample: int) -> np.ndarray:
    if overlay.size == 0:
        return base
    end_sample = start_sample + len(overlay)
    if end_sample > len(base):
        base = np.pad(base, (0, end_sample - len(base)))
    base[start_sample:end_sample] += overlay
    return base


def find_speaker_audio(diarization_dir: Path) -> List[Tuple[str, Path]]:
    speakers = []
    for spk_dir in sorted(diarization_dir.glob("speaker_*")):
        if not spk_dir.is_dir():
            continue
        merged = spk_dir / f"{spk_dir.name}_combined_merged.wav"
        combined = spk_dir / f"{spk_dir.name}_combined.wav"
        if merged.exists():
            speakers.append((spk_dir.name, merged))
        elif combined.exists():
            speakers.append((spk_dir.name, combined))
    return speakers


def collect_separation_outputs(separation_dir: Path) -> Dict[int, List[Path]]:
    mapping: Dict[int, List[Path]] = {}
    for segment_dir in separation_dir.glob("segment*"):
        if not segment_dir.is_dir():
            continue
        name = segment_dir.name
        if not name.startswith("segment"):
            continue
        try:
            idx = int(name.replace("segment", ""))
        except ValueError:
            continue
        voices = sorted(segment_dir.glob("voice*.wav"))
        if voices:
            mapping[idx] = voices
    return mapping


def synthesize_speaker_segments(
    speaker_audio_path: Path,
    segments: List[Dict],
    target_language: str,
    ref_audio: str | None,
    ref_text: str | None,
    tts_model,
    segment_cache_dir: Path | None = None,
) -> Tuple[np.ndarray, List[Dict]]:
    base_audio, sr = load_mono(speaker_audio_path, sr=DEFAULT_SR)
    dubbed = np.zeros_like(base_audio)
    segment_outputs = []

    if not ref_audio:
        ref_audio = str(speaker_audio_path)
    if not ref_text:
        ref_text = ""

    for seg_idx, seg in enumerate(segments, start=1):
        text = seg.get("text", "").strip()
        if not text:
            continue

        translated = seg.get("translated_text", "").strip()
        if not translated:
            translated = translate_fragment(text, target_language)
        if not translated:
            continue
        print(f"[TTS] Segment {seg_idx}: {seg['start']:.2f}-{seg['end']:.2f}s")
        if not ref_text:
            continue

        segment_cache_path = None
        segment_raw_path = None
        if segment_cache_dir is not None:
            segment_cache_dir.mkdir(parents=True, exist_ok=True)
            segment_cache_path = segment_cache_dir / f"segment_{seg_idx:04d}_stretched.wav"
            segment_raw_path = segment_cache_dir / f"segment_{seg_idx:04d}_raw.wav"
            
            if segment_cache_path.exists():
                cached_audio, _ = load_mono(segment_cache_path, sr=DEFAULT_SR)
                dubbed = overlay_audio(dubbed, cached_audio, int(float(seg["start"]) * DEFAULT_SR))
                continue

        if segment_raw_path is not None and segment_raw_path.exists():
            print(f"[TTS] Loaded raw generation from cache for Segment {seg_idx}")
            tts_audio, tts_sr = load_mono(segment_raw_path, sr=DEFAULT_SR)
            wavs = [tts_audio]
        else:
            tts_start = time.time()
            wavs, tts_sr = generate_voice_clone(
                text=translated,
                language=target_language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                model=tts_model,
            )
            print(f"[TTS] Generation time: {time.time() - tts_start:.2f}s")
            
        if not wavs:
            continue

        tts_audio = wavs[0]
        if isinstance(tts_audio, np.ndarray) and tts_audio.ndim > 1:
            tts_audio = tts_audio.mean(axis=1)

        if tts_sr != DEFAULT_SR:
            tts_audio = librosa.resample(tts_audio.astype(np.float32), orig_sr=tts_sr, target_sr=DEFAULT_SR)
            
        if segment_raw_path is not None and not segment_raw_path.exists():
            save_wav(segment_raw_path, tts_audio, DEFAULT_SR)

        target_duration = float(seg["end"] - seg["start"])
        original_duration = len(tts_audio) / DEFAULT_SR
        
        # Don't stretch if it would make it sound completely garbled (e.g. > 2.5x speed or < 0.4x speed)
        rate = original_duration / target_duration if target_duration > 0 else 1.0
        
        if 0.4 <= rate <= 2.5:
            print(f"[TTS] Time-stretch to {target_duration:.2f}s (rate: {rate:.2f}x)")
            tts_audio = time_stretch_to_duration(tts_audio.astype(np.float32), DEFAULT_SR, target_duration)
        else:
            print(f"[WARN] Skipping extreme time-stretch to {target_duration:.2f}s (rate: {rate:.2f}x bounds exceeded)")

        start_sample = int(float(seg["start"]) * DEFAULT_SR)
        dubbed = overlay_audio(dubbed, tts_audio, start_sample)

        if segment_cache_path is not None:
            save_wav(segment_cache_path, tts_audio, DEFAULT_SR)

        segment_outputs.append(
            {
                "segment_index": seg_idx,
                "start": seg["start"],
                "end": seg["end"],
                "source_text": text,
                "translated_text": translated,
            }
        )

    return dubbed, segment_outputs


def mix_audio_tracks(tracks: List[np.ndarray]) -> np.ndarray:
    if not tracks:
        return np.array([], dtype=np.float32)

    max_len = max(len(t) for t in tracks)
    mixed = np.zeros(max_len, dtype=np.float32)
    for t in tracks:
        if len(t) < max_len:
            t = np.pad(t, (0, max_len - len(t)))
        mixed += t

    peak = np.max(np.abs(mixed)) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Movie dubbing pipeline")
    parser.add_argument("--input-audio", required=True, help="Path to input mp3 or wav")
    parser.add_argument("--target-language", default="Chinese", help="Target translation and TTS language")
    parser.add_argument("--hf-token", default="", help="Hugging Face token for overlap detection")
    parser.add_argument("--temp-dir", default=DEFAULT_TEMP_DIR, help="Temporary directory for outputs")
    args = parser.parse_args()

    input_path = Path(args.input_audio).resolve()
    temp_dir = Path(args.temp_dir).resolve()
    output_dir = Path.cwd().resolve()
    cache_dir = temp_dir / DEFAULT_CACHE_DIR

    temp_dir.mkdir(parents=True, exist_ok=True)

    wav_path = ensure_wav(input_path, temp_dir)

    vocal_path = temp_dir / "vocal" / "vocal.wav"
    music_path = temp_dir / "music" / "music.wav"
    if vocal_path.exists() and music_path.exists():
        print("[1/7] Using cached vocal/music outputs...")
    else:
        print("[1/7] Separating vocals and music...")
        vocal_music_separator(
            str(wav_path),
            vocal_dir=str(temp_dir / "vocal"),
            music_dir=str(temp_dir / "music"),
            model_name=DEFAULT_SEPARATOR_MODEL,
        )
    if not vocal_path.exists():
        raise FileNotFoundError(f"Vocal output not found: {vocal_path}")

    env_path = Path.cwd() / ".env"
    hf_token = args.hf_token.strip() if args.hf_token else ""
    if not hf_token:
        hf_token = load_env_value("hf_token", env_path) or ""
    if not hf_token:
        raise RuntimeError(
            "Missing Hugging Face token for overlap detection. "
            "Set hf_token in .env or pass --hf-token."
        )

    overlaps_cache = cache_dir / "overlaps.json"
    if overlaps_cache.exists():
        overlaps = load_json(overlaps_cache, [])
        print(f"[2/7] Using cached overlaps ({len(overlaps)} segments)")
    else:
        print("[2/7] Detecting overlapping speech...")
        overlaps = detect_overlaps(
            hf_token=hf_token,
            audio_file=str(vocal_path),
            plot=False,
        )
        save_json(overlaps_cache, overlaps)

    diarization_dir = temp_dir / "diarization"
    speaker_files = find_speaker_audio(diarization_dir)
    if speaker_files:
        print("[3/7] Using cached diarization outputs...")
    else:
        print("[3/7] Running diarization...")
        perform_diarization_and_extract(
            audio_path=str(vocal_path),
            output_base_dir=str(diarization_dir),
            remove_segments=overlaps if overlaps else None,
        )
        speaker_files = find_speaker_audio(diarization_dir)

    if overlaps:
        separation_dir = temp_dir / "separation"
        separation_done = cache_dir / "separation_done.json"
        if separation_done.exists():
            print("[4/7] Using cached separation outputs...")
        else:
            print("[4/7] Separating overlapping speakers...")
            separate_speakers(
                audio_file=str(vocal_path),
                output_dir=str(separation_dir),
                segments=overlaps,
                model=DEFAULT_SEPARATION_MODEL,
                device=DEFAULT_SEPARATION_DEVICE,
            )
            save_json(separation_done, {"status": "ok"})

        match_done = cache_dir / "match_done.json"
        if match_done.exists():
            print("[5/7] Using cached speaker matches...")
        else:
            print("[5/7] Matching separated voices to diarization speakers...")
            separation_mapping = collect_separation_outputs(separation_dir)
            for seg_idx, voices in separation_mapping.items():
                if seg_idx - 1 >= len(overlaps):
                    continue
                segment = overlaps[seg_idx - 1]
                for voice_path in voices:
                    result = match_and_merge_speaker(
                        input_audio_path=str(voice_path),
                        segment=segment,
                        diarization_dir=str(diarization_dir),
                        threshold=DEFAULT_MATCH_THRESHOLD,
                    )
                    print(result)
            save_json(match_done, {"status": "ok"})
    else:
        print("[4/7] Skipping separation/identification (no overlap segments).")

    print("[6/7] Running ASR per speaker...")
    if not speaker_files:
        raise RuntimeError("No speaker audio files found after diarization.")

    dubbed_tracks = []
    translated_cache_dir = cache_dir / "translations"
    translated_cache_dir.mkdir(parents=True, exist_ok=True)

    print("[6/7] Translating all segments...")
    for speaker_name, speaker_audio_path in speaker_files:
        asr_cache = cache_dir / f"asr_{speaker_name}.json"
        translation_cache = translated_cache_dir / f"translated_{speaker_name}.json"

        if asr_cache.exists():
            print(f"  ASR -> {speaker_name} (cached)")
            result = load_json(asr_cache, {})
        else:
            print(f"  ASR -> {speaker_name}")
            result = transcribe_audio(
                str(speaker_audio_path),
                whisper_model_id=WHISPER_ID,
                device=None,
            )
            save_json(asr_cache, result)

        segments = result.get("segments", [])
        if not segments:
            continue

        if translation_cache.exists():
            print(f"  LLM -> {speaker_name} (cached)")
        else:
            print(f"  LLM -> {speaker_name}")
            translated_segments = []
            for seg_idx, seg in enumerate(segments, start=1):
                text = seg.get("text", "").strip()
                if not text:
                    continue
                print(f"[LLM] Segment {seg_idx}: {seg['start']:.2f}-{seg['end']:.2f}s")
                translated_text = translate_fragment(text, args.target_language)
                translated_segments.append({
                    **seg,
                    "translated_text": translated_text,
                })
            save_json(translation_cache, translated_segments)

    print("[7/7] Running TTS per speaker...")
    tts_model = load_tts_model()
    for speaker_name, speaker_audio_path in speaker_files:
        translation_cache = translated_cache_dir / f"translated_{speaker_name}.json"
        tts_cache = cache_dir / f"tts_{speaker_name}.wav"
        segment_cache_dir = cache_dir / "tts_segments" / speaker_name
        ref_audio_dir = cache_dir / "ref_audio"
        ref_audio_path = ref_audio_dir / f"{speaker_name}_ref_7s.wav"

        translated_segments = load_json(translation_cache, [])
        if not translated_segments:
            continue

        ref_audio_file, ref_text = get_tts_reference(
            speaker_audio_path=speaker_audio_path,
            segments=translated_segments,
            output_ref_path=ref_audio_path,
            target_duration=7.0
        )

        if tts_cache.exists():
            print(f"  TTS -> {speaker_name} (cached)")
            dubbed_audio, _ = load_mono(tts_cache, sr=DEFAULT_SR)
        else:
            print(f"  TTS -> {speaker_name}")
            dubbed_audio, _segment_outputs = synthesize_speaker_segments(
                speaker_audio_path=speaker_audio_path,
                segments=translated_segments,
                target_language=args.target_language,
                ref_audio=ref_audio_file,
                ref_text=ref_text,
                tts_model=tts_model,
                segment_cache_dir=segment_cache_dir,
            )
            save_wav(tts_cache, dubbed_audio, DEFAULT_SR)

        dubbed_tracks.append(dubbed_audio)

    print("[8/8] Mixing music and dubbed speakers...")
    tracks = []
    if music_path.exists():
        music_audio, _ = load_mono(music_path, sr=DEFAULT_SR)
        tracks.append(music_audio)

    tracks.extend(dubbed_tracks)
    final_mix = mix_audio_tracks(tracks)
    final_path = output_dir / "final_mix.wav"
    save_wav(final_path, final_mix, DEFAULT_SR)

    print(f"Done. Final mix: {final_path}")


if __name__ == "__main__":
    main()
