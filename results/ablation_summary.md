# SarcasTone — Ablation Study Results

| Run | Modalities | Fusion | Accuracy | Precision | Recall | F1 | Best-Val-F1 | Best-Epoch | Time (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text_only | text | pgin | 0.6442 | 0.6027 | 0.8462 | 0.7040 | 0.6880 | 3 | 40.1 |
| speech_only | speech | pgin | 0.6250 | 0.5823 | 0.8846 | 0.7023 | 0.6562 | 4 | 29.5 |
| video_only | video | pgin | 0.6923 | 0.7083 | 0.6538 | 0.6800 | 0.7451 | 18 | 62.0 |
| text_speech_concat | text|speech | concat | 0.6538 | 0.6053 | 0.8846 | 0.7188 | 0.7000 | 7 | 36.8 |
| text_speech_pgin | text|speech | pgin | 0.6442 | 0.6056 | 0.8269 | 0.6992 | 0.6667 | 10 | 45.1 |
| text_video | text|video | pgin | 0.7019 | 0.6842 | 0.7500 | 0.7156 | 0.7692 | 14 | 55.0 |
| speech_video | speech|video | pgin | 0.6923 | 0.6923 | 0.6923 | 0.6923 | 0.7636 | 23 | 79.7 |
| full_multimodal_pgin | text|speech|video | pgin | 0.7212 | 0.6716 | 0.8654 | 0.7563 | 0.7458 | 2 | 27.8 |

> Gate weight columns: `gate_text`, `gate_speech`, `gate_inc`, `gate_video`
> are written to `ablation_summary.csv` and `ablation_summary.json`.
