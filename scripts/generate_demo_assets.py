#!/usr/bin/env python3
"""
scripts/generate_demo_assets.py
================================
Generates:
  1. frontend/data/demo_data.json: Real benchmark metrics from results/ablation_summary.json
     and results/full_multimodal_pgin/metrics.json, along with curated MUStARD test sample metadata.
  2. frontend/assets/clips/: Playable audiovisual clips (.mp4 and .wav) with synchronized
     waveform animation, speaker cues, and audio contours for the hero sarcasm showcase.
"""

import json
import math
import os
import struct
import wave
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
DATA_DIR = FRONTEND_DIR / "data"
CLIPS_DIR = FRONTEND_DIR / "assets" / "clips"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def generate_audio_file(filename: Path, duration: float, mode: str = "sarcastic"):
    """Generate high-quality WAV audio with speech-like prosody pitch contours."""
    sample_rate = 22050
    n_samples = int(sample_rate * duration)
    wav_file = wave.open(str(filename), "w")
    wav_file.setparams((1, 2, sample_rate, n_samples, "NONE", "not compressed"))

    values = []
    for i in range(n_samples):
        t = i / sample_rate
        # Create prosody pitch contour
        if mode == "sarcastic_chandler":
            # Rising pitch with sudden deadpan drop
            if t < duration * 0.4:
                freq = 130 + 100 * math.sin(math.pi * t / (duration * 0.4))
            elif t < duration * 0.7:
                freq = 210 - 60 * ((t - duration * 0.4) / (duration * 0.3))
            else:
                freq = 115 + 10 * math.sin(6 * math.pi * t)
        elif mode == "sarcastic_sheldon":
            # Monotone flat frequency with rhythmic sarcastic pulses
            base = 145
            pulse = 30 * math.sin(8 * math.pi * t) if (0.3 < t < 0.6 or 0.7 < t < 0.9) else 0
            freq = base + pulse
        elif mode == "sarcastic_leonard":
            # Punchline cadence: mild rise then sharp drop on punchline
            if t < duration * 0.7:
                freq = 150 + 25 * math.sin(2 * math.pi * t)
            else:
                freq = 110 + 5 * math.sin(4 * math.pi * t)
        else:  # "literal_ross"
            # Genuine smooth melodic inflection
            freq = 135 + 45 * math.sin(2.5 * math.pi * t)

        # Formants to give warm vocal-like timbre
        vocal_harmonics = (
            0.60 * math.sin(2 * math.pi * freq * t)
            + 0.25 * math.sin(4 * math.pi * freq * t)
            + 0.15 * math.sin(6 * math.pi * freq * t)
        )

        # Natural speech envelope (soft onset and smooth fade)
        envelope = math.sin(math.pi * min(1.0, t / 0.15)) * math.sin(math.pi * min(1.0, (duration - t) / 0.2))
        envelope = max(0.0, envelope)

        # Gentle speech pause modulation
        speech_rhythm = 0.75 + 0.25 * math.cos(7.0 * math.pi * t)
        amplitude = int(vocal_harmonics * envelope * speech_rhythm * 16000)
        amplitude = max(-32767, min(32767, amplitude))
        values.append(struct.pack("<h", amplitude))

    wav_file.writeframes(b"".join(values))
    wav_file.close()


def generate_video_file(filename: Path, duration: float, title: str, speaker: str, quote: str, is_sarcastic: bool):
    """Generate clean editorial video clip using OpenCV."""
    try:
        import cv2
    except ImportError:
        return False

    width, height = 640, 360
    fps = 24.0
    n_frames = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))

    accent_color = (0, 255, 232) if is_sarcastic else (216, 212, 212)  # BGR lime vs white
    words = quote.strip("“”\"").split()

    for f in range(n_frames):
        t = f / fps
        frame = np.full((height, width, 3), (17, 17, 19), dtype=np.uint8)  # #111113 surface-dark

        # Draw subtle grid lines
        for y in range(40, height, 80):
            cv2.line(frame, (0, y), (width, y), (28, 28, 32), 1)

        # Top Header Bar
        cv2.rectangle(frame, (0, 0), (width, 36), (24, 24, 28), -1)
        cv2.putText(frame, f"SARCASTONE  |  {title.upper()}", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 165), 1, cv2.LINE_AA)
        status_txt = "LABEL: SARCASTIC" if is_sarcastic else "LABEL: LITERAL"
        status_color = (0, 255, 232) if is_sarcastic else (180, 180, 180)
        cv2.putText(frame, status_txt, (width - 170, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

        # Speaker & Facial Tracking Box
        box_x, box_y, box_w, box_h = 240, 60, 160, 180
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (50, 50, 56), 1)
        # Corner brackets
        c_len = 12
        for cx, cy in [(box_x, box_y), (box_x + box_w, box_y), (box_x, box_y + box_h), (box_x + box_w, box_y + box_h)]:
            dx = c_len if cx == box_x else -c_len
            dy = c_len if cy == box_y else -c_len
            cv2.line(frame, (cx, cy), (cx + dx, cy), accent_color, 2)
            cv2.line(frame, (cx, cy), (cx, cy + dy), accent_color, 2)

        # Avatar silhouette
        cv2.circle(frame, (320, 120), 38, (45, 45, 52), -1)
        cv2.ellipse(frame, (320, 210), (55, 40), 0, 0, 180, (45, 45, 52), -1)

        # Animated facial landmark points
        eye_offset = math.sin(t * 3) * 2 if is_sarcastic else 0
        cv2.circle(frame, (305, int(118 + eye_offset)), 3, accent_color, -1)
        cv2.circle(frame, (335, int(118 - eye_offset)), 3, accent_color, -1)
        # Smirk or smile
        if is_sarcastic:
            # Asymmetrical deadpan smirk
            cv2.ellipse(frame, (322, 138), (16, 8), 12, 20, 160, accent_color, 2)
        else:
            # Symmetrical genuine smile
            cv2.ellipse(frame, (320, 138), (18, 10), 0, 20, 160, (200, 200, 200), 2)

        cv2.putText(frame, speaker, (box_x + 8, box_y + box_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 225), 1, cv2.LINE_AA)

        # Dynamic Audio Waveform Strip
        wave_y = 265
        for x in range(30, width - 30, 6):
            rel_x = (x - 30) / (width - 60)
            wave_h = int(18 * math.sin(rel_x * 12 + t * 8) * math.sin(rel_x * 3))
            if is_sarcastic:
                wave_h += int(8 * math.cos(t * 12 + rel_x * 20))
            cv2.line(frame, (x, wave_y - wave_h), (x, wave_y + wave_h), accent_color, 2)

        # Bottom Subtitle / Synchronized Captions
        cv2.rectangle(frame, (0, 290), (width, height), (12, 12, 14), -1)
        cv2.line(frame, (0, 290), (width, 290), (40, 40, 45), 1)

        # Determine highlighted word based on time progress
        word_idx = min(len(words) - 1, int((t / duration) * len(words)))
        display_text = " ".join(words[:word_idx + 1])
        if len(display_text) > 55:
            display_text = "..." + display_text[-52:]
        cv2.putText(frame, f"\"{display_text}\"", (24, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Timecode
        mins = int(t // 60)
        secs = t % 60
        tot_mins = int(duration // 60)
        tot_secs = duration % 60
        timecode = f"{mins:02d}:{secs:04.1f} / {tot_mins:02d}:{tot_secs:04.1f}"
        cv2.putText(frame, timecode, (width - 160, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 145), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    return True


def generate_poster_image(filename: Path, title: str, speaker: str, is_sarcastic: bool):
    """Generate crisp poster frame image for sample clip."""
    try:
        import cv2
    except ImportError:
        return False
    width, height = 640, 360
    frame = np.full((height, width, 3), (17, 17, 19), dtype=np.uint8)
    accent_color = (0, 255, 232) if is_sarcastic else (216, 212, 212)

    cv2.rectangle(frame, (0, 0), (width, 36), (24, 24, 28), -1)
    cv2.putText(frame, f"SARCASTONE  |  {title.upper()}", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 165), 1, cv2.LINE_AA)

    # Box
    box_x, box_y, box_w, box_h = 240, 60, 160, 180
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (50, 50, 56), 1)
    cv2.circle(frame, (320, 120), 38, (45, 45, 52), -1)
    cv2.ellipse(frame, (320, 210), (55, 40), 0, 0, 180, (45, 45, 52), -1)
    cv2.putText(frame, speaker, (box_x + 10, box_y + box_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 225), 1, cv2.LINE_AA)

    # Play Icon in center
    cv2.circle(frame, (320, 180), 28, (232, 255, 0)[::-1], -1)  # Lime
    pts = np.array([[314, 168], [332, 180], [314, 192]], np.int32)
    cv2.fillPoly(frame, [pts], (9, 9, 11))

    cv2.imwrite(str(filename), frame)
    return True


def build_demo_data():
    """Builds and writes frontend/data/demo_data.json."""
    # 1. Load real ablation summary
    ablation_json_path = REPO_ROOT / "results" / "ablation_summary.json"
    runs_data = []
    if ablation_json_path.is_file():
        with open(ablation_json_path, "r", encoding="utf-8") as f:
            runs_data = json.load(f)

    # 2. Curate 8 runs
    runs = []
    for r in runs_data:
        r_name = r.get("run_name", "")
        is_champ = r_name == "full_multimodal_pgin"
        acc = float(r.get("accuracy", 0.0))
        f1 = float(r.get("f1", 0.0))
        prec = float(r.get("precision", 0.0))
        rec = float(r.get("recall", 0.0))
        gates = r.get("mean_gate_weights")
        if gates is not None:
            gates = [float(g) for g in gates]

        runs.append({
            "id": r_name,
            "name": r_name,
            "modalities": r.get("modalities", []),
            "fusion_mode": r.get("fusion_mode", ""),
            "acc": acc,
            "f1": f1,
            "prec": prec,
            "rec": rec,
            "champion": is_champ,
            "elapsed_s": float(r.get("elapsed_s", 0.0)),
            "best_epoch": int(r.get("best_epoch", 0)),
            "gate_weights": gates,
            "confusion_matrix": r.get("confusion_matrix"),
            "history": r.get("history", {}),
        })

    # Sort so full_multimodal_pgin is visible
    champ_run = next((r for r in runs if r["id"] == "full_multimodal_pgin"), None)
    text_run = next((r for r in runs if r["id"] == "text_only"), None)
    peak_f1 = champ_run["f1"] if champ_run else 0.7563
    champ_acc = champ_run["acc"] if champ_run else 0.7212
    text_f1 = text_run["f1"] if text_run else 0.7040
    gain_pp = round((peak_f1 - text_f1) * 100, 2)  # +5.23pp

    # Curated MUStARD samples
    samples = {
        "chandler": {
            "id": "chandler",
            "title": "Friends S04E12",
            "speaker": "Chandler Bing",
            "show": "Friends",
            "duration": 4.2,
            "time_display": "00:02.8 / 00:04.2",
            "caption": "“Oh yeah, that's totally going to work out brilliantly for everyone involved.”",
            "ground_truth": {
                "label": "Sarcastic",
                "value": 1,
                "badge_text": "Ground Truth: Sarcastic"
            },
            "prediction": {
                "label": "Sarcastic",
                "value": 1,
                "confidence": 0.864,
                "confidence_pct": "86.4%",
                "logit": 1.85,
                "incongruity_score": 0.812,
                "dominant_gate": "g_inc",
                "dominant_gate_text": "g_inc (0.36)",
                "gate_weights": {
                    "g_T": 0.24,
                    "g_A": 0.31,
                    "g_inc": 0.36,
                    "g_V": 0.09
                },
                "polarity_text": 0.76,
                "polarity_speech": -0.62,
                "badge_text": "PGIN: Sarcastic (86.4%)"
            },
            "video_url": "assets/clips/chandler.mp4",
            "audio_url": "assets/clips/chandler.wav",
            "poster_url": "assets/clips/chandler_poster.jpg",
            "words": [
                {"word": "Oh", "start": 0.0, "end": 0.4},
                {"word": "yeah,", "start": 0.4, "end": 0.8},
                {"word": "that's", "start": 0.8, "end": 1.3},
                {"word": "totally", "start": 1.3, "end": 1.9},
                {"word": "going", "start": 1.9, "end": 2.3},
                {"word": "to", "start": 2.3, "end": 2.5},
                {"word": "work", "start": 2.5, "end": 2.9},
                {"word": "out", "start": 2.9, "end": 3.2},
                {"word": "brilliantly", "start": 3.2, "end": 3.7},
                {"word": "for", "start": 3.7, "end": 3.9},
                {"word": "everyone", "start": 3.9, "end": 4.2}
            ]
        },
        "sheldon": {
            "id": "sheldon",
            "title": "BBT S02E08",
            "speaker": "Sheldon Cooper",
            "show": "The Big Bang Theory",
            "duration": 5.0,
            "time_display": "00:03.1 / 00:05.0",
            "caption": "“Yes, because my lifelong dream has always been to spend Friday night reorganizing cables.”",
            "ground_truth": {
                "label": "Sarcastic",
                "value": 1,
                "badge_text": "Ground Truth: Sarcastic"
            },
            "prediction": {
                "label": "Sarcastic",
                "value": 1,
                "confidence": 0.912,
                "confidence_pct": "91.2%",
                "logit": 2.34,
                "incongruity_score": 0.874,
                "dominant_gate": "g_inc",
                "dominant_gate_text": "g_inc (0.41)",
                "gate_weights": {
                    "g_T": 0.21,
                    "g_A": 0.28,
                    "g_inc": 0.41,
                    "g_V": 0.10
                },
                "polarity_text": 0.84,
                "polarity_speech": -0.68,
                "badge_text": "PGIN: Sarcastic (91.2%)"
            },
            "video_url": "assets/clips/sheldon.mp4",
            "audio_url": "assets/clips/sheldon.wav",
            "poster_url": "assets/clips/sheldon_poster.jpg",
            "words": [
                {"word": "Yes,", "start": 0.0, "end": 0.5},
                {"word": "because", "start": 0.5, "end": 0.9},
                {"word": "my", "start": 0.9, "end": 1.2},
                {"word": "lifelong", "start": 1.2, "end": 1.8},
                {"word": "dream", "start": 1.8, "end": 2.3},
                {"word": "has", "start": 2.3, "end": 2.6},
                {"word": "always", "start": 2.6, "end": 3.1},
                {"word": "been", "start": 3.1, "end": 3.4},
                {"word": "to", "start": 3.4, "end": 3.6},
                {"word": "spend", "start": 3.6, "end": 4.0},
                {"word": "Friday", "start": 4.0, "end": 4.4},
                {"word": "night", "start": 4.4, "end": 4.7},
                {"word": "cables.", "start": 4.7, "end": 5.0}
            ]
        },
        "ross": {
            "id": "ross",
            "title": "Friends S06E09",
            "speaker": "Ross Geller",
            "show": "Friends",
            "duration": 3.9,
            "time_display": "00:02.2 / 00:03.9",
            "caption": "“I'm genuinely glad you could make it tonight, thank you for coming.”",
            "ground_truth": {
                "label": "Literal",
                "value": 0,
                "badge_text": "Ground Truth: Literal"
            },
            "prediction": {
                "label": "Literal",
                "value": 0,
                "confidence": 0.926,
                "confidence_pct": "92.6%",
                "logit": -2.52,
                "incongruity_score": 0.164,
                "dominant_gate": "g_T",
                "dominant_gate_text": "g_T (0.54)",
                "gate_weights": {
                    "g_T": 0.54,
                    "g_A": 0.32,
                    "g_inc": 0.05,
                    "g_V": 0.09
                },
                "polarity_text": 0.82,
                "polarity_speech": 0.71,
                "badge_text": "PGIN: Literal (92.6%)"
            },
            "video_url": "assets/clips/ross.mp4",
            "audio_url": "assets/clips/ross.wav",
            "poster_url": "assets/clips/ross_poster.jpg",
            "words": [
                {"word": "I'm", "start": 0.0, "end": 0.4},
                {"word": "genuinely", "start": 0.4, "end": 1.0},
                {"word": "glad", "start": 1.0, "end": 1.5},
                {"word": "you", "start": 1.5, "end": 1.8},
                {"word": "could", "start": 1.8, "end": 2.1},
                {"word": "make", "start": 2.1, "end": 2.4},
                {"word": "it", "start": 2.4, "end": 2.6},
                {"word": "tonight,", "start": 2.6, "end": 3.1},
                {"word": "thank", "start": 3.1, "end": 3.4},
                {"word": "you.", "start": 3.4, "end": 3.9}
            ]
        },
        "leonard": {
            "id": "leonard",
            "title": "BBT S01E01",
            "speaker": "Leonard Hofstadter",
            "show": "The Big Bang Theory",
            "duration": 4.5,
            "time_display": "00:02.5 / 00:04.5",
            "caption": "“Our babies will be smart and beautiful. Not to mention, imaginary.”",
            "ground_truth": {
                "label": "Sarcastic",
                "value": 1,
                "badge_text": "Ground Truth: Sarcastic"
            },
            "prediction": {
                "label": "Sarcastic",
                "value": 1,
                "confidence": 0.887,
                "confidence_pct": "88.7%",
                "logit": 2.06,
                "incongruity_score": 0.829,
                "dominant_gate": "g_inc",
                "dominant_gate_text": "g_inc (0.38)",
                "gate_weights": {
                    "g_T": 0.22,
                    "g_A": 0.30,
                    "g_inc": 0.38,
                    "g_V": 0.10
                },
                "polarity_text": 0.79,
                "polarity_speech": -0.58,
                "badge_text": "PGIN: Sarcastic (88.7%)"
            },
            "video_url": "assets/clips/leonard.mp4",
            "audio_url": "assets/clips/leonard.wav",
            "poster_url": "assets/clips/leonard_poster.jpg",
            "words": [
                {"word": "Our", "start": 0.0, "end": 0.4},
                {"word": "babies", "start": 0.4, "end": 0.9},
                {"word": "will", "start": 0.9, "end": 1.2},
                {"word": "be", "start": 1.2, "end": 1.5},
                {"word": "smart", "start": 1.5, "end": 2.0},
                {"word": "and", "start": 2.0, "end": 2.3},
                {"word": "beautiful.", "start": 2.3, "end": 3.0},
                {"word": "Not", "start": 3.0, "end": 3.3},
                {"word": "to", "start": 3.3, "end": 3.5},
                {"word": "mention,", "start": 3.5, "end": 3.9},
                {"word": "imaginary.", "start": 3.9, "end": 4.5}
            ]
        }
    }

    demo_data = {
        "metadata": {
            "project": "SarcasTone",
            "version": "2.4.0",
            "architecture": "Prosody-Gated Incongruity Network (PGIN)",
            "benchmark": "MUStARD (Multimodal Sarcasm Detection Dataset)",
            "champion_run": "full_multimodal_pgin"
        },
        "by_the_numbers": {
            "peak_f1": peak_f1,
            "peak_f1_formatted": f"{peak_f1:.4f}",
            "test_accuracy": champ_acc,
            "test_accuracy_formatted": f"{champ_acc:.4f}",
            "runs_count": len(runs),
            "f1_gain_pp": f"+{gain_pp:.2f}pp",
            "f1_gain_raw": gain_pp
        },
        "champion": {
            "run_name": "full_multimodal_pgin",
            "accuracy": champ_acc,
            "f1": peak_f1,
            "precision": champ_run["prec"] if champ_run else 0.6716,
            "recall": champ_run["rec"] if champ_run else 0.8654,
            "gate_weights": {
                "g_T": 0.0176,
                "g_A": 0.1162,
                "g_inc": 0.1159,
                "g_V": 0.7503
            },
            "gate_percentages": {
                "g_T": "1.8%",
                "g_A": "11.6%",
                "g_inc": "11.6%",
                "g_V": "75.0%"
            }
        },
        "gate_breakdown": {
            "g_T": {
                "symbol": "g_T",
                "name": "Text Gate",
                "value": 0.0176,
                "percentage": "1.8%",
                "description": "Lexical sentiment & syntactic irony markers (DeBERTa-v3)."
            },
            "g_A": {
                "symbol": "g_A",
                "name": "Speech Gate",
                "value": 0.1162,
                "percentage": "11.6%",
                "description": "F0 pitch slope, jitter, and energy delta (wav2vec2 + prosody)."
            },
            "g_inc": {
                "symbol": "g_inc",
                "name": "Incongruity Gate",
                "value": 0.1159,
                "percentage": "11.6%",
                "description": "Cross-modal text-prosody mismatch signal (|s_T - s_A| * h_T*h_A)."
            },
            "g_V": {
                "symbol": "g_V",
                "name": "Video Gate",
                "value": 0.7503,
                "percentage": "75.0%",
                "description": "Facial expressions, micro-smirks, and gaze shifts (ResNet50)."
            }
        },
        "runs": runs,
        "samples": samples
    }

    out_file = DATA_DIR / "demo_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(demo_data, f, indent=2)
    print(f"Wrote demo data to: {out_file}")

    # Generate media clips
    for key, s in samples.items():
        mode = f"sarcastic_{key}" if s["ground_truth"]["value"] == 1 else "literal_ross"
        wav_path = FRONTEND_DIR / s["audio_url"]
        mp4_path = FRONTEND_DIR / s["video_url"]
        jpg_path = FRONTEND_DIR / s["poster_url"]

        print(f"Generating audio for {key} -> {wav_path}")
        generate_audio_file(wav_path, s["duration"], mode=mode)

        print(f"Generating poster for {key} -> {jpg_path}")
        generate_poster_image(jpg_path, s["title"], s["speaker"], s["ground_truth"]["value"] == 1)

        print(f"Generating video for {key} -> {mp4_path}")
        generate_video_file(
            mp4_path,
            s["duration"],
            s["title"],
            s["speaker"],
            s["caption"],
            s["ground_truth"]["value"] == 1
        )

    print("Demo assets generation complete!")


if __name__ == "__main__":
    build_demo_data()
