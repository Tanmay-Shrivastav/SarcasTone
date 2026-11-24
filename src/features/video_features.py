"""
Video visual feature extraction using ResNet-50 for SarcasTone.
Uniformly samples video frames across utterance clips and extracts
temporally-averaged 2048-dimensional visual representations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision import transforms
    TORCH_VISION_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_VISION_AVAILABLE = False

logger = logging.getLogger(__name__)

VIDEO_FEAT_DIM = 2048

# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sample_frames_from_video(
    video_path: Union[str, Path],
    num_frames: int = 8,
    frame_size: int = 224,
) -> Tuple[np.ndarray, bool]:
    """
    Uniformly sample num_frames RGB frames across a video clip.
    Resizes each frame to (frame_size, frame_size) and applies ImageNet normalization.

    Returns:
        Tuple of (frames_array of shape (num_frames, 3, frame_size, frame_size), is_valid boolean).
    """
    p = Path(video_path)
    empty_frames = np.zeros((num_frames, 3, frame_size, frame_size), dtype=np.float32)

    if not p.is_file():
        logger.warning(f"Video file not found: {video_path}")
        return empty_frames, False

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        logger.warning(f"Failed to open video file: {video_path}")
        return empty_frames, False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Attempt sequential read if frame count metadata is missing
        raw_frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            raw_frames.append(frame)
        cap.release()

        if not raw_frames:
            logger.warning(f"No readable frames found in {video_path}")
            return empty_frames, False

        total_frames = len(raw_frames)
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        sampled_bgr = [raw_frames[idx] for idx in frame_indices]
    else:
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        sampled_bgr = []
        last_valid_frame = None

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                sampled_bgr.append(frame)
                last_valid_frame = frame
            elif last_valid_frame is not None:
                # Pad corrupted/missing frame with last valid frame
                sampled_bgr.append(last_valid_frame)
            else:
                # Seek first readable frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_first, frame_first = cap.read()
                if ret_first and frame_first is not None:
                    sampled_bgr.append(frame_first)
                    last_valid_frame = frame_first
        cap.release()

    if not sampled_bgr:
        logger.warning(f"Could not extract any valid frames from {video_path}")
        return empty_frames, False

    # Pad if fewer frames were extracted than num_frames
    while len(sampled_bgr) < num_frames:
        sampled_bgr.append(sampled_bgr[-1] if sampled_bgr else np.zeros((frame_size, frame_size, 3), dtype=np.uint8))

    processed_frames = np.zeros((num_frames, 3, frame_size, frame_size), dtype=np.float32)

    for i, frame in enumerate(sampled_bgr[:num_frames]):
        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        # Scale to [0, 1]
        norm = resized.astype(np.float32) / 255.0
        # ImageNet normalization
        norm = (norm - IMAGENET_MEAN) / IMAGENET_STD
        # Transpose from (H, W, C) to (C, H, W)
        processed_frames[i] = np.transpose(norm, (2, 0, 1))

    return processed_frames, True


class VideoFeatureExtractor:
    """
    Extracts 2048-dimensional visual representations from video clips using ResNet-50.
    Samples frames uniformly across the utterance and averages pooled frame features.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        num_frames: int = 8,
        frame_size: int = 224,
        device: Optional[str] = None,
        use_amp: bool = True,
    ):
        self.model_name = model_name
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.use_amp = use_amp

        if device is None:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.backbone = None
        self._is_initialized = False

    def _initialize_model(self) -> None:
        """Lazily initialize ResNet-50 backbone without classification head."""
        if self._is_initialized:
            return

        if not TORCH_VISION_AVAILABLE:
            raise RuntimeError(
                "torch and torchvision are required to extract visual features via ResNet50. "
                "Please run on Google Colab or install requirements.txt."
            )

        logger.info(f"Loading visual encoder ({self.model_name}) on {self.device}...")
        weights = models.ResNet50_Weights.DEFAULT
        full_resnet = models.resnet50(weights=weights)

        # Truncate final FC classification layer; keep global average pooling
        layers = list(full_resnet.children())[:-1]  # Output shape: (B, 2048, 1, 1)
        self.backbone = nn.Sequential(*layers)
        self.backbone.to(self.device)
        self.backbone.eval()
        self._is_initialized = True
        logger.info("Visual backbone loaded successfully.")

    def extract_from_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Pass sampled normalized frames through ResNet-50 and temporally average.

        Args:
            frames: Array of shape (num_frames, 3, frame_size, frame_size).

        Returns:
            np.ndarray of shape (2048,) with float32 values.
        """
        self._initialize_model()

        tensor_frames = torch.from_numpy(frames).to(self.device)

        with torch.no_grad():
            if self.use_amp and self.device == "cuda":
                with torch.amp.autocast("cuda"):
                    features = self.backbone(tensor_frames)  # (N, 2048, 1, 1)
            else:
                features = self.backbone(tensor_frames)

            # Flatten to (N, 2048) and temporally average across frames
            features = features.view(features.size(0), -1)  # (num_frames, 2048)
            pooled = torch.mean(features, dim=0)            # (2048,)
            return pooled.cpu().float().numpy().astype(np.float32)

    def extract_from_file(
        self,
        video_path: Optional[Union[str, Path]],
    ) -> Tuple[np.ndarray, bool]:
        """
        Extract visual representation from a video clip file.

        Returns:
            Tuple of (video_emb of shape (2048,), video_valid boolean).
        """
        if not video_path:
            return np.zeros((VIDEO_FEAT_DIM,), dtype=np.float32), False

        frames, valid = sample_frames_from_video(
            video_path=video_path,
            num_frames=self.num_frames,
            frame_size=self.frame_size,
        )

        if not valid:
            return np.zeros((VIDEO_FEAT_DIM,), dtype=np.float32), False

        try:
            video_emb = self.extract_from_frames(frames)
            return video_emb, True
        except Exception as e:
            logger.warning(f"Error extracting visual features from {video_path}: {e}")
            return np.zeros((VIDEO_FEAT_DIM,), dtype=np.float32), False
