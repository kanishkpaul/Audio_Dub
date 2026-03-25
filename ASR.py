import threading
import torch
import numpy as np
import librosa
import soundfile as sf
import string
from typing import List, Dict, Optional, Tuple, Any, Iterator
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

TARGET_SR  = 16_000
WHISPER_ID = "openai/whisper-small"
HINDI_WHISPER_ID = "vasista22/whisper-hindi-small"
MAX_WHISPER_CHUNK_SEC = 25.0
WHISPER_CHUNK_OVERLAP_SEC = 2.0
MAX_WHISPER_BATCH_WINDOWS = 6
MIN_LID_CHUNK_SEC = 1.0
DEBUG_ASR = True

# ── Language boundary detection settings ─────────────────────────────────────
# How large each probe window is (seconds). Smaller = finer splits, but noisier.
LBD_WINDOW_SEC   = 3.0
# How far to advance the probe each step (seconds). Should be < LBD_WINDOW_SEC.
LBD_STEP_SEC     = 1.5
# Only attempt splitting on chunks longer than this (seconds).
LBD_MIN_CHUNK_SEC = 6.0
# Minimum sub-segment length after a split (seconds). Avoids hair-thin slivers.
LBD_MIN_SPLIT_SEC = 2.0
# How many consecutive windows must agree on a *different* language before we
# commit to a split.  Raises robustness against single-window misdetections.
LBD_CONFIRM_WINDOWS = 2
# ─────────────────────────────────────────────────────────────────────────────

_WHISPER_CACHE: Dict[Tuple[str, str], Tuple[Any, ...]] = {}
_WHISPER_CACHE_LOCK = threading.Lock()
_WHISPER_LOADING: Dict[Tuple[str, str], threading.Event] = {}

_VAD_CACHE: Optional[Tuple[Any, Any]] = None
_VAD_CACHE_LOCK = threading.Lock()
_VAD_LOAD_FAILURES = 0

CODESWITCHING_PAIRS = {
    frozenset({"hi", "en"}),
    frozenset({"ur", "en"}),
    frozenset({"es", "en"}),
    frozenset({"ar", "en"}),
    frozenset({"zh", "en"}),
    frozenset({"fr", "en"}),
}


def _normalize_word(w: str) -> str:
    return w.strip(string.punctuation).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_mono_16k(path: str) -> Tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(path, always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio file: {path}") from exc

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    mono = audio.mean(axis=1).astype(np.float32)
    if mono.size == 0:
        raise ValueError(f"Audio file contains no samples after mono conversion: {path}")

    if sr != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    return mono, sr


# ─────────────────────────────────────────────────────────────────────────────
# VAD
# ─────────────────────────────────────────────────────────────────────────────

def run_vad(audio: np.ndarray, min_speech_ms: int = 200, min_silence_ms: int = 500) -> List[Dict]:
    global _VAD_CACHE, _VAD_LOAD_FAILURES
    if _VAD_CACHE is None:
        with _VAD_CACHE_LOCK:
            if _VAD_CACHE is None:
                try:
                    _VAD_CACHE = torch.hub.load(
                        "snakers4/silero-vad", "silero_vad", trust_repo=True
                    )
                    if _VAD_LOAD_FAILURES > 0:
                        print(f"Silero VAD loaded after {_VAD_LOAD_FAILURES} previous failure(s).")
                        _VAD_LOAD_FAILURES = 0
                except Exception as exc:
                    _VAD_LOAD_FAILURES += 1
                    raise RuntimeError("Failed to load Silero VAD model") from exc

    vad_model, utils = _VAD_CACHE
    vad_model.to("cpu").eval()
    get_ts = utils[0]
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to("cpu")
    spans = get_ts(
        wav, vad_model,
        sampling_rate=TARGET_SR,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        return_seconds=True,
    )
    return spans


def _span_to_indices(span: Dict, total_samples: int) -> Tuple[int, int, float, float]:
    start = float(span.get("start", 0.0))
    end   = float(span.get("end",   0.0))
    duration_sec = total_samples / TARGET_SR
    t_start = max(0.0, start)
    t_end   = min(duration_sec, end)
    s = int(round(t_start * TARGET_SR))
    e = int(round(t_end   * TARGET_SR))
    if e < s:
        e = s
    return s, e, t_start, t_end


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_whisper(model_id: str, device: str):
    cache_key = (model_id, device)
    should_load = False
    loading_event: Optional[threading.Event] = None

    with _WHISPER_CACHE_LOCK:
        cached = _WHISPER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        loading_event = _WHISPER_LOADING.get(cache_key)
        if loading_event is None:
            loading_event = threading.Event()
            _WHISPER_LOADING[cache_key] = loading_event
            should_load = True

    if not should_load:
        if loading_event is None:
            raise RuntimeError(f"Whisper model '{model_id}' loading state was missing")
        loading_event.wait()
        with _WHISPER_CACHE_LOCK:
            cached = _WHISPER_CACHE.get(cache_key)
        if cached is None:
            raise RuntimeError(f"Whisper model '{model_id}' failed to load in a concurrent worker")
        return cached

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    try:
        try:
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
            ).to(device).eval()
            print("Using cached Whisper files.")
        except Exception:
            print(f"Downloading {model_id} from Hugging Face...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, dtype=dtype, low_cpu_mem_usage=True,
            ).to(device).eval()

        lang_to_id = getattr(model.generation_config, "lang_to_id", None)
        if isinstance(lang_to_id, dict) and len(lang_to_id) > 0:
            id_to_lang = {v: k.strip("<|>") for k, v in lang_to_id.items()}
            lang_ids   = torch.tensor(sorted(lang_to_id.values()), device=device, dtype=torch.long)
            vocab_proj = (model.proj_out if hasattr(model, "proj_out")
                          else model.get_output_embeddings())
            sot_id     = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        else:
            # Some fine-tuned Whisper checkpoints do not expose multilingual
            # language-token metadata. They can still be used for ASR.
            id_to_lang = None
            lang_ids = None
            vocab_proj = None
            sot_id = None

        loaded = (processor, model, id_to_lang, lang_ids, vocab_proj, sot_id)
        with _WHISPER_CACHE_LOCK:
            _WHISPER_CACHE[cache_key] = loaded
        return loaded
    except Exception as exc:
        raise RuntimeError(f"Failed to load Whisper model '{model_id}'") from exc
    finally:
        with _WHISPER_CACHE_LOCK:
            event = _WHISPER_LOADING.pop(cache_key, None)
            if event is not None:
                event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Language identification
# ─────────────────────────────────────────────────────────────────────────────

def detect_language(
    chunk: np.ndarray,
    processor, model, id_to_lang, lang_ids, vocab_proj, sot_id,
    device: str,
    topk: int = 3,
) -> List[Tuple[str, float]]:
    if any(x is None for x in (id_to_lang, lang_ids, vocab_proj, sot_id)):
        raise RuntimeError("Language detection requires a Whisper model with lang_to_id metadata")

    if len(chunk) == 0:
        return [("en", 0.0)]

    min_len = int(MIN_LID_CHUNK_SEC * TARGET_SR)
    if len(chunk) < min_len:
        chunk = np.pad(chunk, (0, min_len - len(chunk)))

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    feats = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
    inp   = feats.input_features.to(device=device, dtype=dtype)

    with torch.inference_mode():
        enc    = model.model.encoder(inp).last_hidden_state
        dec_in = torch.tensor([[sot_id]], device=device, dtype=torch.long)
        hidden = model.model.decoder(
            input_ids=dec_in,
            encoder_hidden_states=enc,
        ).last_hidden_state[:, -1, :]
        logits      = vocab_proj(hidden)
        lang_logits = logits[0, lang_ids]
        probs       = torch.softmax(lang_logits.float(), dim=-1)

    k   = min(topk, len(probs))
    top = torch.topk(probs, k)
    return [
        (id_to_lang[lang_ids[p].item()], round(s, 4))
        for p, s in zip(top.indices.tolist(), top.values.tolist())
    ]


def resolve_language(
    top_langs: List[Tuple[str, float]],
    prev_lang: Optional[str],
    conf_threshold: float = 0.35,
) -> Tuple[str, float]:
    best_lang, best_score = top_langs[0]

    if best_score >= conf_threshold:
        return best_lang, best_score

    if len(top_langs) >= 2:
        second_lang, second_score = top_langs[1]
        pair = frozenset({best_lang, second_lang})
        if pair in CODESWITCHING_PAIRS:
            dominant = best_lang
            if pair == frozenset({"hi", "ur"}):
                dominant = "hi"
            return dominant, min(1.0, best_score + second_score)

    if prev_lang is not None:
        return prev_lang, best_score

    return best_lang, best_score


# ─────────────────────────────────────────────────────────────────────────────
# ★ NEW: Language Boundary Detection
#
# Slides a probe window across a VAD chunk and looks for a stable language
# change.  Returns a list of (start_sample, end_sample, language, confidence)
# sub-segments.  If no boundary is found the original span is returned intact.
# ─────────────────────────────────────────────────────────────────────────────

def _probe_language_at(
    audio: np.ndarray,
    probe_start_sample: int,
    probe_end_sample: int,
    processor, model, id_to_lang, lang_ids, vocab_proj, sot_id,
    device: str,
) -> str:
    """Return the top-1 language for a short audio probe window."""
    probe = audio[probe_start_sample:probe_end_sample]
    top   = detect_language(
        probe, processor, model, id_to_lang, lang_ids, vocab_proj, sot_id,
        device, topk=1,
    )
    return top[0][0]


def split_chunk_on_language_boundary(
    chunk: np.ndarray,
    chunk_t_start: float,
    processor, model, id_to_lang, lang_ids, vocab_proj, sot_id,
    device: str,
    dominant_lang: str,
) -> List[Dict]:
    """
    Slide a probe window over *chunk* and detect where the language changes
    from *dominant_lang*.  Returns a list of dicts:
        {"audio": np.ndarray, "t_start": float, "t_end": float, "language": str}

    If no boundary is found, returns a single entry covering the whole chunk.
    """
    chunk_dur = len(chunk) / TARGET_SR

    # Not worth splitting very short chunks.
    if chunk_dur < LBD_MIN_CHUNK_SEC:
        return [{"audio": chunk, "t_start": chunk_t_start,
                 "t_end": chunk_t_start + chunk_dur, "language": dominant_lang}]

    win_samples  = int(LBD_WINDOW_SEC * TARGET_SR)
    step_samples = int(LBD_STEP_SEC   * TARGET_SR)
    min_split_samples = int(LBD_MIN_SPLIT_SEC * TARGET_SR)

    # Collect per-window language labels.
    window_langs: List[Tuple[int, int, str]] = []   # (start, end, lang)
    pos = 0
    while pos < len(chunk):
        end = min(pos + win_samples, len(chunk))
        lang = _probe_language_at(
            chunk, pos, end,
            processor, model, id_to_lang, lang_ids, vocab_proj, sot_id, device,
        )
        window_langs.append((pos, end, lang))
        if end >= len(chunk):
            break
        pos += step_samples

    # Find the first position where LBD_CONFIRM_WINDOWS consecutive windows
    # all agree on a language that differs from dominant_lang.
    boundary_sample: Optional[int] = None
    boundary_lang:   Optional[str] = None

    for i in range(len(window_langs) - LBD_CONFIRM_WINDOWS + 1):
        langs_in_window = [window_langs[i + j][2] for j in range(LBD_CONFIRM_WINDOWS)]
        if all(l != dominant_lang for l in langs_in_window) and len(set(langs_in_window)) == 1:
            candidate_boundary = window_langs[i][0]
            # Enforce minimum sub-segment length on both sides.
            if (candidate_boundary >= min_split_samples and
                    len(chunk) - candidate_boundary >= min_split_samples):
                boundary_sample = candidate_boundary
                boundary_lang   = langs_in_window[0]
                break

    if boundary_sample is None:
        # No clean boundary found → return the whole chunk.
        return [{"audio": chunk, "t_start": chunk_t_start,
                 "t_end": chunk_t_start + chunk_dur, "language": dominant_lang}]

    # Split at the detected boundary.
    boundary_t = chunk_t_start + boundary_sample / TARGET_SR
    part_a = chunk[:boundary_sample]
    part_b = chunk[boundary_sample:]

    print(f"  ★ Language boundary detected at {boundary_t:.2f}s "
          f"({dominant_lang} → {boundary_lang}). Splitting chunk.")

    sub_a = {"audio": part_a, "t_start": chunk_t_start,
             "t_end": boundary_t, "language": dominant_lang}
    sub_b = {"audio": part_b, "t_start": boundary_t,
             "t_end": chunk_t_start + chunk_dur, "language": boundary_lang}

    # Recursively check sub_b for further boundaries (handles >2 languages).
    further = split_chunk_on_language_boundary(
        part_b, boundary_t,
        processor, model, id_to_lang, lang_ids, vocab_proj, sot_id,
        device, boundary_lang,
    )
    return [sub_a] + further


# ─────────────────────────────────────────────────────────────────────────────
# ASR helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iter_whisper_windows(chunk: np.ndarray) -> Iterator[np.ndarray]:
    max_len = int(MAX_WHISPER_CHUNK_SEC * TARGET_SR)
    overlap = int(WHISPER_CHUNK_OVERLAP_SEC * TARGET_SR)
    if len(chunk) <= max_len:
        yield chunk
        return
    step  = max(1, max_len - overlap)
    start = 0
    while start < len(chunk):
        end = min(start + max_len, len(chunk))
        yield chunk[start:end]
        if end >= len(chunk):
            break
        start += step


def _merge_window_text(text_parts: List[str], piece: str, max_overlap_words: int = 12) -> None:
    if not piece:
        return
    if not text_parts:
        text_parts.append(piece)
        return
    prev_words = text_parts[-1].split()
    cur_words  = piece.split()
    if not prev_words or not cur_words:
        text_parts.append(piece)
        return
    max_k     = min(max_overlap_words, len(prev_words), len(cur_words))
    overlap_k = 0
    for k in range(max_k, 0, -1):
        if ([_normalize_word(w) for w in prev_words[-k:]] ==
                [_normalize_word(w) for w in cur_words[:k]]):
            overlap_k = k
            break
    if overlap_k > 0:
        remainder = " ".join(cur_words[overlap_k:]).strip()
        if remainder:
            text_parts.append(remainder)
        elif overlap_k == len(cur_words):
            text_parts.append(piece)
    else:
        text_parts.append(piece)


def _transcribe_windows(
    windows: List[np.ndarray],
    language: str,
    processor, model,
    device: str,
    max_new_tokens: int = 444,
) -> List[str]:
    if not windows:
        return []
    dtype   = torch.float16 if device.startswith("cuda") else torch.float32
    outputs: List[str] = []
    has_lang_tokens = hasattr(model.generation_config, "lang_to_id")
    model_name = getattr(model.config, "_name_or_path", "<unknown>")
    if DEBUG_ASR:
        print(f"[DEBUG] _transcribe_windows model={model_name} has_lang_tokens={has_lang_tokens} language={language}")
    for i in range(0, len(windows), MAX_WHISPER_BATCH_WINDOWS):
        batch = windows[i:i + MAX_WHISPER_BATCH_WINDOWS]
        feats = processor(batch, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inp   = feats.input_features.to(device=device, dtype=dtype)
        try:
            with torch.inference_mode():
                gen_kwargs = {"max_new_tokens": max_new_tokens}
                if has_lang_tokens:
                    gen_kwargs["language"] = language
                    gen_kwargs["task"] = "transcribe"
                if DEBUG_ASR:
                    print(f"[DEBUG] batch {i//MAX_WHISPER_BATCH_WINDOWS}: generate kwargs={list(gen_kwargs.keys())}")
                ids = model.generate(inp, **gen_kwargs)
            batch_text = [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]
        except ValueError as exc:
            if "generation config is outdated" in str(exc).lower():
                if DEBUG_ASR:
                    print("[DEBUG] Retrying generate() without language/task due outdated generation_config")
                with torch.inference_mode():
                    ids = model.generate(inp, max_new_tokens=max_new_tokens)
                batch_text = [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]
            else:
                raise
        except RuntimeError as exc:
            is_oom = device.startswith("cuda") and ("out of memory" in str(exc).lower())
            if not is_oom:
                raise
            torch.cuda.empty_cache()
            batch_text = [
                _transcribe_chunk(w, language, processor, model, device, max_new_tokens)
                for w in batch
            ]
        outputs.extend(batch_text)
    return outputs


def _transcribe_chunk(
    chunk: np.ndarray,
    language: str,
    processor, model,
    device: str,
    max_new_tokens: int = 444,
) -> str:
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    feats = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
    inp   = feats.input_features.to(device=device, dtype=dtype)
    has_lang_tokens = hasattr(model.generation_config, "lang_to_id")
    model_name = getattr(model.config, "_name_or_path", "<unknown>")
    if DEBUG_ASR:
        print(f"[DEBUG] _transcribe_chunk model={model_name} has_lang_tokens={has_lang_tokens} language={language}")
    try:
        with torch.inference_mode():
            gen_kwargs = {"max_new_tokens": max_new_tokens}
            if has_lang_tokens:
                gen_kwargs["language"] = language
                gen_kwargs["task"] = "transcribe"
            if DEBUG_ASR:
                print(f"[DEBUG] chunk generate kwargs={list(gen_kwargs.keys())}")
            ids = model.generate(inp, **gen_kwargs)
    except ValueError as exc:
        if "generation config is outdated" in str(exc).lower():
            if DEBUG_ASR:
                print("[DEBUG] Retrying chunk generate() without language/task due outdated generation_config")
            with torch.inference_mode():
                ids = model.generate(inp, max_new_tokens=max_new_tokens)
        else:
            raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device.startswith("cuda"):
            torch.cuda.empty_cache()
        raise
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: str,
    whisper_model_id: str = WHISPER_ID,
    device: Optional[str] = None,
    min_speech_ms: int = 200,
    min_silence_ms: int = 500,
    lid_topk: int = 3,
    lid_conf_threshold: float = 0.35,
    max_new_tokens: int = 444,
) -> Dict:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    audio, sr = load_mono_16k(audio_path)
    print(f"Duration: {len(audio)/TARGET_SR:.2f}s | Original SR: {sr}Hz")

    print("Running VAD...")
    spans = run_vad(audio, min_speech_ms=min_speech_ms, min_silence_ms=min_silence_ms)
    print(f"  {len(spans)} speech chunk(s) from VAD")
    if not spans:
        return {
            "audio_path": audio_path,
            "original_sr": sr,
            "segments": [],
            "warning": "No speech segments detected by VAD.",
        }

    print(f"Loading {whisper_model_id}...")
    processor, model, id_to_lang, lang_ids, vocab_proj, sot_id = load_whisper(
        whisper_model_id, device
    )
    hindi_processor = None
    hindi_model     = None

    segments: List[Dict] = []
    prev_lang = None

    print("Running LID + Language Boundary Detection + ASR...")
    for span in spans:
        s, e, t_start, t_end = _span_to_indices(span, total_samples=len(audio))
        if e <= s:
            continue
        chunk = audio[s:e]
        if len(chunk) == 0:
            continue

        # ── Step 1: Identify dominant language for this VAD span ──────────────
        lid_chunk       = chunk[:min(len(chunk), int(15 * TARGET_SR))]
        lid_short_chunk = (len(lid_chunk) / TARGET_SR) < 3.0
        if lid_short_chunk:
            print(f"  Warning: short chunk ({len(lid_chunk)/TARGET_SR:.2f}s), "
                  "language confidence may be unstable.")

        top_langs      = detect_language(
            lid_chunk, processor, model,
            id_to_lang, lang_ids, vocab_proj, sot_id,
            device, topk=lid_topk,
        )
        dominant_lang, conf = resolve_language(top_langs, prev_lang,
                                               conf_threshold=lid_conf_threshold)

        # ── Step 2: Check for a language boundary inside the chunk ────────────
        sub_segs = split_chunk_on_language_boundary(
            chunk, t_start,
            processor, model, id_to_lang, lang_ids, vocab_proj, sot_id,
            device, dominant_lang,
        )

        # ── Step 3: Transcribe each sub-segment with the right model ──────────
        for sub in sub_segs:
            sub_audio  = sub["audio"]
            sub_lang   = sub["language"]
            sub_t_start = sub["t_start"]
            sub_t_end   = sub["t_end"]
            sub_dur     = sub_t_end - sub_t_start

            # Re-run LID on the sub-segment to get its own confidence score.
            sub_lid = sub_audio[:min(len(sub_audio), int(15 * TARGET_SR))]
            sub_top = detect_language(
                sub_lid, processor, model,
                id_to_lang, lang_ids, vocab_proj, sot_id,
                device, topk=lid_topk,
            )
            _, sub_conf = resolve_language(sub_top, prev_lang,
                                           conf_threshold=lid_conf_threshold)
            prev_lang = sub_lang

            # Pick the right ASR model.
            if sub_lang == "hi":
                if hindi_processor is None or hindi_model is None:
                    print(f"  Loading {HINDI_WHISPER_ID} for Hindi segments...")
                    hindi_processor, hindi_model, *_ = load_whisper(HINDI_WHISPER_ID, device)
                asr_processor = hindi_processor
                asr_model     = hindi_model
            else:
                asr_processor = processor
                asr_model     = model

            windows = list(_iter_whisper_windows(sub_audio))
            pieces  = _transcribe_windows(
                windows, sub_lang, asr_processor, asr_model, device, max_new_tokens,
            )

            text_parts: List[str] = []
            for piece in pieces:
                _merge_window_text(text_parts, piece)

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            text = " ".join(text_parts).strip()

            segments.append({
                "start":          round(sub_t_start, 3),
                "end":            round(sub_t_end,   3),
                "language":       sub_lang,
                "confidence":     round(sub_conf, 4),
                "lid_short_chunk": (sub_dur < 3.0),
                "text":           text,
            })
            print(f"  [{sub_t_start:.2f}-{sub_t_end:.2f}] {sub_lang} "
                  f"({sub_conf:.2f}) | {text}")

    return {
        "audio_path":  audio_path,
        "original_sr": sr,
        "segments":    segments,
    }