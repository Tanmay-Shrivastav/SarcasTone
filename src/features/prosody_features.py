"""
Speech prosody and acoustic feature extraction for SarcasTone.
Extracts:
  1. Explicit 16-dimensional prosody vector (pitch F0, energy RMS, speaking rate, pauses).
  2. 768-dimensional acoustic embeddings via facebook/wav2vec2-base.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None  # type: ignore
    sf = None  # type: ignore
    LIBROSA_AVAILABLE = False

try:
    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    TORCH_AUDIO_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)

PROSODY_DIM = 16
ACOUSTIC_DIM = 768


def compute_linear_slope(values: np.ndarray) -> float:
    """Calculate the linear slope (rate of change) of a 1D sequence."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float32)
    # Mean-center to prevent numerical issues
    x_centered = x - np.mean(x)
    denom = np.sum(x_centered**2)
    if denom <= 1e-9:
        return 0.0
    y_centered = values - np.mean(values)
    slope = float(np.sum(x_centered * y_centered) / denom)
    return float(np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0))


def extract_explicit_prosody(
    audio: np.ndarray,
    sr: int = 16000,
    fmin: float = 65.0,
    fmax: float = 500.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    silence_threshold_db: float = 35.0,
) -> np.ndarray:
    """
    Extract a 16-dimensional explicit prosody vector capturing:
      1. Pitch (F0) dynamics: voiced mean, std, min, max, range, linear slope, voiced ratio (7 dims)
      2. Energy (RMS): mean, std, max, dynamic range, linear slope (5 dims)
      3. Speaking rate & rhythm: onset rate per second, zero crossing rate (2 dims)
      4. Silence & pauses: silent frame ratio, total utterance duration (2 dims)

    Returns:
        np.ndarray of shape (16,) with float32 values. All NaNs/Infs are replaced with 0.0.
    """
    duration_sec = float(len(audio)) / float(sr) if sr > 0 else 0.0
    if len(audio) < hop_length or duration_sec <= 0.01:
        return np.zeros((PROSODY_DIM,), dtype=np.float32)

    # Normalize audio amplitude safely
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio_norm = audio / peak
    else:
        audio_norm = audio

    # --- 1. Pitch / Fundamental Frequency (F0) (7 dims) ---
    f0_mean = 0.0
    f0_std = 0.0
    f0_min = 0.0
    f0_max = 0.0
    f0_range = 0.0
    f0_slope = 0.0
    voiced_ratio = 0.0

    if LIBROSA_AVAILABLE:
        try:
            f0 = librosa.yin(
                audio_norm,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            # Mask out unvoiced frames (YIN returns ~fmin or ~fmax on unvoiced or NaN)
            voiced_mask = (f0 >= fmin * 1.05) & (f0 <= fmax * 0.95) & (~np.isnan(f0))
            total_frames = len(f0)
            if total_frames > 0:
                voiced_ratio = float(np.sum(voiced_mask)) / float(total_frames)

            if np.any(voiced_mask):
                voiced_f0 = f0[voiced_mask]
                f0_mean = float(np.mean(voiced_f0))
                f0_std = float(np.std(voiced_f0))
                f0_min = float(np.min(voiced_f0))
                f0_max = float(np.max(voiced_f0))
                f0_range = f0_max - f0_min
                f0_slope = compute_linear_slope(voiced_f0)
        except Exception as e:
            logger.debug(f"YIN pitch extraction fallback: {e}")
    else:
        # Fallback autocorrelation-based pitch estimation when librosa is absent
        corr = signal.correlate(audio_norm, audio_norm, mode="full")
        corr = corr[len(corr) // 2 :]
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)
        if max_lag < len(corr) and min_lag < max_lag:
            peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
            est_f0 = float(sr / peak_lag) if peak_lag > 0 else 0.0
            f0_mean = est_f0
            f0_min = est_f0
            f0_max = est_f0
            voiced_ratio = 1.0

    # --- 2. Energy / Intensity (RMS) (5 dims) ---
    if LIBROSA_AVAILABLE:
        rms_frames = librosa.feature.rms(y=audio_norm, frame_length=frame_length, hop_length=hop_length)[0]
    else:
        # Pure NumPy framing for RMS
        num_frames = max(1, (len(audio_norm) - frame_length) // hop_length + 1)
        rms_frames = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            frame = audio_norm[i * hop_length : i * hop_length + frame_length]
            rms_frames[i] = np.sqrt(np.mean(frame**2)) if len(frame) > 0 else 0.0

    rms_mean = float(np.mean(rms_frames)) if len(rms_frames) > 0 else 0.0
    rms_std = float(np.std(rms_frames)) if len(rms_frames) > 0 else 0.0
    rms_max = float(np.max(rms_frames)) if len(rms_frames) > 0 else 0.0
    rms_dyn_range = float(rms_max - rms_mean)
    rms_slope = compute_linear_slope(rms_frames)

    # --- 3. Speaking Rate & Rhythm (2 dims) ---
    onset_rate = 0.0
    zcr_mean = 0.0

    if LIBROSA_AVAILABLE:
        try:
            onsets = librosa.onset.onset_detect(y=audio_norm, sr=sr, hop_length=hop_length)
            onset_rate = float(len(onsets)) / duration_sec if duration_sec > 0 else 0.0
        except Exception:
            onset_rate = 0.0

        zcr = librosa.feature.zero_crossing_rate(audio_norm, frame_length=frame_length, hop_length=hop_length)[0]
        zcr_mean = float(np.mean(zcr)) if len(zcr) > 0 else 0.0
    else:
        # Pure NumPy zero crossing rate
        zero_crossings = np.nonzero(np.diff(audio_norm > 0))[0]
        zcr_mean = float(len(zero_crossings)) / float(len(audio_norm)) if len(audio_norm) > 0 else 0.0
        onset_rate = float(len(zero_crossings)) / (duration_sec * 10.0) if duration_sec > 0 else 0.0

    # --- 4. Silence & Pauses (2 dims) ---
    if rms_max > 1e-5:
        # Frame energy in decibels relative to max
        rms_db = 20.0 * np.log10(np.maximum(rms_frames, 1e-9) / rms_max)
        silent_frames = np.sum(rms_db < -silence_threshold_db)
        silence_ratio = float(silent_frames) / float(len(rms_frames)) if len(rms_frames) > 0 else 0.0
    else:
        silence_ratio = 1.0

    prosody_features = [
        f0_mean,
        f0_std,
        f0_min,
        f0_max,
        f0_range,
        f0_slope,
        voiced_ratio,
        rms_mean,
        rms_std,
        rms_max,
        rms_dyn_range,
        rms_slope,
        onset_rate,
        zcr_mean,
        silence_ratio,
        duration_sec,
    ]

    prosody_vec = np.nan_to_num(np.array(prosody_features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return prosody_vec


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".flv"}


def extract_audio_from_media(
    media_path: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, bool]:
    """
    Extract single-channel (mono) audio array at target_sr from audio or video media.
    Supports both audio files (.wav, .mp3, .flac) and video containers (.mp4, .avi, .mkv, .mov).

    Extraction pipeline:
      1. If media is a video format (e.g. .mp4), uses ffmpeg CLI (if available)
         to stream decoded 16 kHz mono PCM WAV directly into memory via pipe:1.
      2. Torchaudio decoder fallback (supports .mp4 via FFmpeg/TorchCodec backend).
      3. Librosa loader (audioread backend).
      4. Soundfile / scipy.io.wavfile (for standard audio containers).
      5. PyAV (av) or moviepy if installed.

    Returns:
        Tuple of (audio_array: np.ndarray [float32], is_valid: bool).
    """
    p = Path(media_path)
    if not p.is_file():
        logger.warning(f"Media file not found: {media_path}")
        return np.zeros(target_sr, dtype=np.float32), False

    ext = p.suffix.lower()

    # 1. FFmpeg CLI: fast, zero-disk overhead, in-memory stream decoding
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-nostdin",
                "-loglevel", "error",
                "-i", str(p.resolve()),
                "-vn",                  # disable video stream
                "-ac", "1",             # mono
                "-ar", str(target_sr),  # target sample rate (16000 Hz)
                "-f", "wav",
                "pipe:1",               # output to stdout
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            if proc.returncode == 0 and len(proc.stdout) > 44:
                try:
                    sr_read, raw_audio = wavfile.read(io.BytesIO(proc.stdout))
                    if raw_audio.ndim > 1:
                        raw_audio = np.mean(raw_audio, axis=-1)

                    if raw_audio.dtype == np.int16:
                        audio = raw_audio.astype(np.float32) / 32768.0
                    elif raw_audio.dtype == np.int32:
                        audio = raw_audio.astype(np.float32) / 2147483648.0
                    elif raw_audio.dtype == np.uint8:
                        audio = (raw_audio.astype(np.float32) - 128.0) / 128.0
                    else:
                        audio = raw_audio.astype(np.float32)

                    if sr_read != target_sr and sr_read > 0:
                        num_samples = int(len(audio) * target_sr / sr_read)
                        audio = signal.resample(audio, num_samples)

                    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                    if len(audio) > 0:
                        return audio.astype(np.float32), True
                except Exception as parse_err:
                    logger.debug(f"wavfile parse failed on ffmpeg output for {media_path}: {parse_err}")
        except Exception as ffmpeg_err:
            logger.debug(f"ffmpeg subprocess execution failed for {media_path}: {ffmpeg_err}")

    # 2. Torchaudio loader fallback (PyTorch torchaudio decodes video containers if ffmpeg backend present)
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(p))
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != target_sr:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
        audio = waveform.squeeze().cpu().numpy().astype(np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if len(audio) > 0:
            return audio, True
    except Exception as e:
        logger.debug(f"torchaudio.load failed for {media_path}: {e}")

    # 3. Librosa loader (works for audio files and video files if audioread has ffmpeg backend)
    if LIBROSA_AVAILABLE:
        try:
            audio, sr = librosa.load(str(p), sr=target_sr, mono=True)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            if len(audio) > 0:
                return audio.astype(np.float32), True
        except Exception as e:
            logger.debug(f"librosa.load failed for {media_path}: {e}")

    # 4. Soundfile loader (standard for audio files: .wav, .flac, .ogg)
    if sf is not None and ext not in VIDEO_EXTENSIONS:
        try:
            audio, sr = sf.read(str(p))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != target_sr and sr > 0:
                num_samples = int(len(audio) * target_sr / sr)
                audio = signal.resample(audio, num_samples)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            if len(audio) > 0:
                return audio.astype(np.float32), True
        except Exception as e:
            logger.debug(f"soundfile.read failed for {media_path}: {e}")

    # 5. Scipy wavfile loader (pure fallback for .wav without librosa/soundfile)
    if ext == ".wav":
        try:
            sr, raw_audio = wavfile.read(str(p))
            if raw_audio.ndim > 1:
                raw_audio = np.mean(raw_audio, axis=-1)
            if raw_audio.dtype == np.int16:
                audio = raw_audio.astype(np.float32) / 32768.0
            elif raw_audio.dtype == np.int32:
                audio = raw_audio.astype(np.float32) / 2147483648.0
            else:
                audio = raw_audio.astype(np.float32)
            if sr != target_sr and sr > 0:
                num_samples = int(len(audio) * target_sr / sr)
                audio = signal.resample(audio, num_samples)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            return audio.astype(np.float32), True
        except Exception as e:
            logger.debug(f"scipy.io.wavfile failed for {media_path}: {e}")

    # 6. PyAV (av) decoder fallback
    try:
        import av
        container = av.open(str(p))
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is not None:
            frames = []
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            for frame in container.decode(audio_stream):
                for resampled_frame in resampler.resample(frame):
                    frames.append(resampled_frame.to_ndarray()[0])
            if frames:
                audio = np.concatenate(frames).astype(np.float32)
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                return audio, True
    except Exception as e:
        logger.debug(f"PyAV audio extraction failed for {media_path}: {e}")

    # 7. MoviePy decoder fallback
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(p))
        if clip.audio is not None:
            audio_arr = clip.audio.to_soundarray(fps=target_sr)
            if audio_arr.ndim > 1:
                audio = np.mean(audio_arr, axis=1)
            else:
                audio = audio_arr
            clip.close()
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            return audio.astype(np.float32), True
    except Exception as e:
        logger.debug(f"moviepy audio extraction failed for {media_path}: {e}")

    logger.warning(f"Failed to decode audio track from {media_path} with any available backend.")
    return np.zeros(target_sr, dtype=np.float32), False


def extract_wav_from_video(
    video_path: Union[str, Path],
    output_wav_path: Union[str, Path],
    target_sr: int = 16000,
) -> bool:
    """
    Helper function to extract 16 kHz mono audio from a video file and save to a .wav file.
    Useful for offline pre-extraction or caching audio tracks to disk.
    """
    audio, valid = extract_audio_from_media(video_path, target_sr=target_sr)
    if not valid:
        return False
    try:
        out_p = Path(output_wav_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        int_audio = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
        wavfile.write(str(out_p), target_sr, int_audio)
        return True
    except Exception as e:
        logger.error(f"Failed to save extracted wav to {output_wav_path}: {e}")
        return False


class ProsodyAcousticExtractor:
    """
    Extracts joint speech representations for an utterance:
      1. 16-dim explicit prosody vector.
      2. 768-dim Wav2Vec2 acoustic embedding.
    """

    def __init__(
        self,
        audio_model: str = "facebook/wav2vec2-base",
        sample_rate: int = 16000,
        max_audio_sec: float = 15.0,
        fmin: float = 65.0,
        fmax: float = 500.0,
        frame_length: int = 2048,
        hop_length: int = 512,
        silence_threshold_db: float = 35.0,
        device: Optional[str] = None,
        use_amp: bool = True,
    ):
        self.audio_model = audio_model
        self.sample_rate = sample_rate
        self.max_audio_sec = max_audio_sec
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.silence_threshold_db = silence_threshold_db
        self.use_amp = use_amp

        if device is None:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.processor = None
        self.wav2vec_model = None
        self._is_initialized = False

    def _initialize_model(self) -> None:
        """Lazily initialize Wav2Vec2 acoustic backbone on device."""
        if self._is_initialized:
            return

        if not TORCH_AUDIO_AVAILABLE:
            raise RuntimeError(
                "transformers and torch are required to extract Wav2Vec2 features. "
                "Please run on Google Colab or install requirements.txt."
            )

        logger.info(f"Loading speech acoustic backbone: {self.audio_model} on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.audio_model)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(self.audio_model)
        self.wav2vec_model.to(self.device)
        self.wav2vec_model.eval()
        self._is_initialized = True
        logger.info("Speech acoustic backbone loaded successfully.")

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, bool]:
        """
        Load audio, convert to mono, resample to target sample_rate, and cap duration.
        Automatically handles .mp4 video clips by extracting the embedded audio stream.

        Returns:
            (audio_array, is_valid_boolean)
        """
        p = Path(audio_path)
        if not p.is_file():
            logger.warning(f"Audio file not found: {audio_path}")
            return np.zeros(self.sample_rate, dtype=np.float32), False

        audio, valid = extract_audio_from_media(p, target_sr=self.sample_rate)
        if not valid:
            return np.zeros(self.sample_rate, dtype=np.float32), False

        # Cap maximum audio duration to prevent memory spikes on Colab T4
        max_samples = int(self.sample_rate * self.max_audio_sec)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        return audio.astype(np.float32), True

    def extract_acoustic_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract 768-dimensional mean-pooled Wav2Vec2 acoustic representation.
        """
        if len(audio) == 0:
            return np.zeros((ACOUSTIC_DIM,), dtype=np.float32)

        self._initialize_model()

        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            if self.use_amp and self.device == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = self.wav2vec_model(input_values)
            else:
                outputs = self.wav2vec_model(input_values)

            # Mean-pool over time frames
            last_hidden = outputs.last_hidden_state  # (1, T, 768)
            pooled = torch.mean(last_hidden, dim=1).squeeze(0)  # (768,)
            return pooled.cpu().float().numpy().astype(np.float32)

    def extract_from_file(
        self,
        audio_path: Optional[Union[str, Path]],
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Extract both explicit prosody and acoustic embeddings for an audio file.

        Returns:
            Tuple of (prosody_vec of shape (16,), audio_emb of shape (768,), audio_valid boolean)
        """
        if not audio_path:
            return (
                np.zeros((PROSODY_DIM,), dtype=np.float32),
                np.zeros((ACOUSTIC_DIM,), dtype=np.float32),
                False,
            )

        audio, valid = self.load_audio(audio_path)
        if not valid:
            return (
                np.zeros((PROSODY_DIM,), dtype=np.float32),
                np.zeros((ACOUSTIC_DIM,), dtype=np.float32),
                False,
            )

        prosody_vec = extract_explicit_prosody(
            audio=audio,
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            silence_threshold_db=self.silence_threshold_db,
        )

        try:
            audio_emb = self.extract_acoustic_embedding(audio)
        except Exception as e:
            logger.warning(f"Error extracting acoustic embedding for {audio_path}: {e}")
            audio_emb = np.zeros((ACOUSTIC_DIM,), dtype=np.float32)

        return prosody_vec, audio_emb, True
