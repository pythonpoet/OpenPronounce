import numpy as np
import torchaudio
import torch
import audio
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
import Levenshtein
import re
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
import hashlib
import functools
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# 1. TORCH CPU OPTIMISATIONS
#    - Set number of threads to match Koyeb eco-xlarge (4 vCPU)
#    - Enable oneDNN (MKL-DNN) fusion for conv/linear ops
# ---------------------------------------------------------------------------
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.quantized.engine = "qnnpack"  # lighter quantization backend for CPU


# ---------------------------------------------------------------------------
# 2. MODEL LOADING — load once, quantize, set to eval
#    INT8 dynamic quantization cuts model size ~50% and speeds up linear
#    layers (the dominant op in Wav2Vec2) by 1.5–2× on CPU with no
#    meaningful accuracy loss for inference.
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-dutch"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

_model_fp32 = Wav2Vec2Model.from_pretrained(MODEL_NAME)
_model_fp32.eval()
model = torch.quantization.quantize_dynamic(
    _model_fp32,
    {torch.nn.Linear},  # only quantize Linear layers
    dtype=torch.qint8,
)
del _model_fp32  # free FP32 weights immediately

_modelCTC_fp32 = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
_modelCTC_fp32.eval()
modelCTC = torch.quantization.quantize_dynamic(
    _modelCTC_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8,
)
del _modelCTC_fp32


# ---------------------------------------------------------------------------
# 3. PHONEMIZER — reuse a single backend instance instead of spawning
#    a new espeak subprocess on every call. EspeakBackend is thread-safe
#    for read-only phonemization.
# ---------------------------------------------------------------------------
_espeak_backend = EspeakBackend(
    language="nl",
    preserve_punctuation=False,
    with_stress=False,
)


def _phonemize_word(word: str) -> list[str]:
    """Phonemize a single word using the shared espeak backend."""
    try:
        result = _espeak_backend.phonemize([word], separator=None)
        return result[0].split() if result else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 4. CACHES
#    - LRU cache for phoneme sequences (text is repeated across requests)
#    - LRU cache for TTS reference embeddings (same text → same reference)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=512)
def _cached_word_phonemes(word: str) -> tuple[str, ...]:
    """Cache phoneme list per word. Returns a tuple (hashable)."""
    return tuple(_phonemize_word(word))


@functools.lru_cache(maxsize=64)
def _cached_reference_embeddings(text: str) -> np.ndarray:
    """
    Cache TTS + Wav2Vec2 embeddings for a reference text.
    Keyed by text string. Saves 2 expensive ops on repeated exercises.
    """
    reference_file = audio.text2speech(text)
    audio_2, sr = torchaudio.load(reference_file)
    return extract_embeddings(audio_2.squeeze().numpy(), sr)


@functools.lru_cache(maxsize=512)
def _cached_phoneme_sequence(text: str) -> tuple[list, dict]:
    """Cache the full phoneme sequence for a reference text."""
    return get_phonemes_with_word_mapping(text)


# ---------------------------------------------------------------------------
# 5. PARALLEL INFERENCE
#    Run the three forward passes (user embeddings, reference embeddings,
#    transcription) concurrently using threads. The GIL is released during
#    torch C++ ops, so this gives real parallelism on multi-core CPUs.
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=3)


def _run_parallel_inference(audio_1, text_reference, sampling_rate):
    """
    Submit all three inference tasks simultaneously and collect results.
    Wall-clock time ≈ slowest single task instead of sum of all three.
    """
    fut_emb_user = _executor.submit(extract_embeddings, audio_1, sampling_rate)
    fut_emb_ref  = _executor.submit(_cached_reference_embeddings, text_reference)
    fut_transcribe = _executor.submit(transcribe, audio_1)

    emb_1 = fut_emb_user.result()
    emb_2 = fut_emb_ref.result()
    transcription = fut_transcribe.result()

    return emb_1, emb_2, transcription


# ---------------------------------------------------------------------------
# 6. CORE FUNCTIONS (optimised)
# ---------------------------------------------------------------------------

def extract_embeddings(audio_waveform, sampling_rate=16000):
    """Extract Wav2Vec2 embeddings. Uses quantized model."""
    inputs = processor(
        audio_waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values
    if len(input_values.shape) > 2:
        input_values = input_values.squeeze(0)

    with torch.no_grad():
        # torch.inference_mode is stricter than no_grad and slightly faster
        with torch.inference_mode():
            features = model(input_values).last_hidden_state

    return features.squeeze(0).numpy()


def get_phonemes_with_word_mapping(text: str):
    """Return phoneme list and word mapping. Uses per-word LRU cache."""
    words = re.findall(r"\b[\w']+\b", text)
    phonemes = []
    phoneme_to_word = {}

    for word in words:
        word_phonemes = list(_cached_word_phonemes(word.lower()))
        for phoneme in word_phonemes:
            phoneme_to_word[len(phonemes)] = word
            phonemes.append(phoneme)

    return phonemes, phoneme_to_word


def transcribe(audio_input) -> str:
    """Transcribe audio to text using quantized CTC model."""
    inputs = processor(
        audio_input,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    with torch.inference_mode():
        logits = modelCTC(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


def get_phoneme_embeddings(phoneme_seq):
    return np.array([ord(p) for p in phoneme_seq], dtype=np.float32).reshape(-1, 1)


def align_sequences_dtw(seq1, seq2):
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    aligned_seq1 = [seq1[i][0] for i, _ in path]
    aligned_seq2 = [seq2[j][0] for _, j in path]
    return np.array(aligned_seq1), np.array(aligned_seq2)


def compare_transcriptions(transcription: str, text_reference: str):
    transcription_clean = transcription.lower().strip()
    reference_clean = text_reference.lower().strip()

    word_distance = Levenshtein.distance(transcription_clean, reference_clean)

    # Use cached phoneme sequences
    expected_phonemes, expected_map = _cached_phoneme_sequence(text_reference)
    transcribed_phonemes, transcribed_map = get_phonemes_with_word_mapping(transcription_clean)

    expected_seq = get_phoneme_embeddings(" ".join(expected_phonemes))
    transcribed_seq = get_phoneme_embeddings(" ".join(transcribed_phonemes))

    distance, _ = fastdtw(expected_seq, transcribed_seq, dist=euclidean)

    errors = []
    words_with_errors = set()
    alignment_map = [set() for _ in range(len(expected_phonemes))]

    opcodes = Levenshtein.opcodes(expected_phonemes, transcribed_phonemes)
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k, l in zip(range(i1, i2), range(j1, j2)):
                alignment_map[k].add(l)
        elif tag == "replace":
            len_i, len_j = i2 - i1, j2 - j1
            for k in range(i1, i2):
                start_j = j1 + int((k - i1) * len_j / len_i)
                end_j   = j1 + int((k - i1 + 1) * len_j / len_i)
                if start_j == end_j and len_j > 0:
                    alignment_map[k].add(min(start_j, j2 - 1))
                else:
                    for l in range(start_j, end_j):
                        alignment_map[k].add(l)

    expected_words_list = re.findall(r"\b[\w']+\b", text_reference)
    current_phoneme_idx = 0

    for word in expected_words_list:
        p_list = list(_cached_word_phonemes(word.lower()))
        if not p_list:
            continue

        word_indices = range(current_phoneme_idx, current_phoneme_idx + len(p_list))
        current_phoneme_idx += len(p_list)

        matched_trans_indices = set()
        for idx in word_indices:
            if idx < len(alignment_map):
                matched_trans_indices.update(alignment_map[idx])

        if not matched_trans_indices:
            errors.append({"position": word_indices.start, "expected": word, "actual": "", "word": word})
            words_with_errors.add(word)
        else:
            sorted_trans_indices = sorted(matched_trans_indices)
            seen_words, actual_words = set(), []
            for tidx in sorted_trans_indices:
                if tidx in transcribed_map:
                    w = transcribed_map[tidx]
                    if w not in seen_words:
                        actual_words.append(w)
                        seen_words.add(w)

            actual_text = " ".join(actual_words)
            expected_seg = [expected_phonemes[i] for i in word_indices]
            actual_seg   = [transcribed_phonemes[i] for i in sorted_trans_indices]
            p_dist = Levenshtein.distance(expected_seg, actual_seg)

            if p_dist > len(expected_seg) * 0.4:
                errors.append({
                    "position": word_indices.start,
                    "expected": "".join(expected_seg),
                    "actual":   "".join(actual_seg),
                    "word":     word,
                    "actual_word": actual_text,
                })
                words_with_errors.add(word)

    feedback = "🔊 Feedback on your pronunciation:\n"
    if words_with_errors:
        feedback += "❌ You need to better pronounce these words: " + ", ".join(words_with_errors) + "\n"
    else:
        feedback += "✅ Your pronunciation is excellent! 🎉\n"

    expected_vector = expected_seq.tolist()
    transcribed_vector = transcribed_seq.tolist()
    expected_vector, transcribed_vector = align_sequences_dtw(expected_vector, transcribed_vector)

    return {
        "word_distance": word_distance,
        "phoneme_distance": distance,
        "errors": errors,
        "feedback": feedback,
        "transcribe": transcription,
        "expected_vector": expected_vector.astype(float).tolist(),
        "transcribed_vector": transcribed_vector.astype(float).tolist(),
        "expected_phonemes": expected_phonemes,
        "transcribed_phonemes": transcribed_phonemes,
        "words_with_errors": list(words_with_errors),
    }


def compute_pronunciation_score(distance_dtw, phoneme_distance, word_distance, max_dtw=500, max_lev=30):
    dtw_score     = max(0, 100 - (distance_dtw      / max_dtw) * 100)
    phoneme_score = max(0, 100 - (phoneme_distance  / max_dtw) * 100)
    word_score    = max(0, 100 - (word_distance      / max_lev) * 100)
    final_score   = 0.4 * dtw_score + 0.3 * phoneme_score + 0.3 * word_score
    return round(max(0.0, min(100.0, final_score)), 2)


def extract_f0(audio_waveform, sr=16000):
    f0, _, _ = librosa.pyin(audio_waveform, fmin=50, fmax=300)
    return np.nan_to_num(f0)


def extract_energy(audio_waveform):
    energy = librosa.feature.rms(y=audio_waveform)
    scaler = MinMaxScaler(feature_range=(0, 250))
    return scaler.fit_transform(energy.T).flatten()


def interpolate_f0(f0):
    f0 = np.array(f0)
    mask = f0 > 0
    return np.interp(np.arange(len(f0)), np.where(mask)[0], f0[mask])


def compare_audio_with_text(audio_1, text_reference, sampling_rate=16000):
    """
    Main entry point. Runs 3 inference tasks in parallel, then
    combines results. Prosody extraction runs after, also in parallel.
    """
    # --- parallel: embeddings + transcription ---
    emb_1, emb_2, transcription = _run_parallel_inference(
        audio_1, text_reference, sampling_rate
    )

    # --- parallel: DTW + prosody (both CPU/numpy, no GIL contention) ---
    fut_dtw     = _executor.submit(fastdtw, emb_1, emb_2, euclidean)
    fut_energy  = _executor.submit(extract_energy, audio_1)
    fut_f0      = _executor.submit(extract_f0, audio_1, sampling_rate)

    distance, _ = fut_dtw.result()
    distance = int(distance)

    differences = compare_transcriptions(transcription, text_reference)
    score = compute_pronunciation_score(
        distance,
        differences["phoneme_distance"],
        differences["word_distance"],
    )

    energy = fut_energy.result()
    f0     = interpolate_f0(fut_f0.result())

    return {
        "score": score,
        "distance": distance,
        "differences": differences,
        "feedback": differences["feedback"],
        "transcribe": differences["transcribe"],
        "prosody": {
            "f0": f0.tolist(),
            "energy": energy.tolist(),
        },
    }
