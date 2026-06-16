#!/usr/bin/env python3
"""
Load and visualize camera frames saved by recorder_plugin HDF5 files.

Examples (from repo root):
  python3 scripts/h5_camera_dataloader.py --list
  python3 scripts/h5_camera_dataloader.py --index 0 --no-gui
  python3 scripts/h5_camera_dataloader.py --play --fps 15
  python3 scripts/h5_camera_dataloader.py --rgb-key kinec2/rgb --depth-key kinec2/depth
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np


DEFAULT_H5 = "data/recording_20260616_183722_1781606242460.h5"


@dataclass(frozen=True)
class CameraFrame:
    key: str
    group: str
    index: int
    timestamp_ns: Optional[int]
    width: int
    height: int
    channels: int
    format_type: int
    mode: str
    image: np.ndarray


class H5CameraDataLoader:
    """Read recorder_plugin camera frames from HDF5 image blob groups."""

    REQUIRED_DATASETS = (
        "width",
        "height",
        "channels",
        "format_type",
        "image_data",
        "image_offsets",
        "image_lengths",
    )

    def __init__(self, h5_path: str | Path) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

    @staticmethod
    def key_to_group(key: str) -> str:
        return key.strip("/").replace("/", "_")

    def list_groups(self) -> Dict[str, Dict[str, Tuple[Tuple[int, ...], str]]]:
        """Return all HDF5 groups with their dataset shapes and dtypes."""
        groups: Dict[str, Dict[str, Tuple[Tuple[int, ...], str]]] = {}
        with h5py.File(self.h5_path, "r") as h5:
            for group_name, obj in h5.items():
                if not isinstance(obj, h5py.Group):
                    continue
                datasets: Dict[str, Tuple[Tuple[int, ...], str]] = {}
                for name, ds in obj.items():
                    if isinstance(ds, h5py.Dataset):
                        datasets[name] = (tuple(ds.shape), str(ds.dtype))
                groups[group_name] = datasets
        return groups

    def available_camera_groups(self) -> Iterable[str]:
        groups = self.list_groups()
        for group_name, datasets in groups.items():
            if all(name in datasets for name in self.REQUIRED_DATASETS):
                yield group_name

    def frame_count(self, key: str) -> int:
        group_name = self.key_to_group(key)
        with h5py.File(self.h5_path, "r") as h5:
            group = self._require_group(h5, group_name)
            self._require_camera_group(group, group_name)
            return int(group["image_lengths"].shape[0])

    def read_pair(
        self, rgb_key: str = "realsense/rgb", depth_key: str = "realsense/depth", index: int = 0
    ) -> Tuple[CameraFrame, CameraFrame]:
        return self.read_frame(rgb_key, index), self.read_frame(depth_key, index)

    def read_frame(self, key: str, index: int = 0) -> CameraFrame:
        group_name = self.key_to_group(key)
        with h5py.File(self.h5_path, "r") as h5:
            group = self._require_group(h5, group_name)
            self._require_camera_group(group, group_name)

            frame_count = int(group["image_lengths"].shape[0])
            if frame_count == 0:
                raise IndexError(f"Group '{group_name}' contains no frames")
            if index < 0:
                index += frame_count
            if index < 0 or index >= frame_count:
                raise IndexError(f"Frame index {index} out of range for '{group_name}' ({frame_count} frames)")

            width = int(group["width"][index])
            height = int(group["height"][index])
            channels = int(group["channels"][index])
            format_type = int(group["format_type"][index])
            offset = int(group["image_offsets"][index])
            length = int(group["image_lengths"][index])
            timestamp_ns = int(group["timestamps_ns"][index]) if "timestamps_ns" in group else None

            blob = group["image_data"][offset : offset + length]
            payload = np.asarray(blob).view(np.uint8)
            image, mode = self._decode_image(payload, width, height, channels, format_type, key)

        return CameraFrame(
            key=key,
            group=group_name,
            index=index,
            timestamp_ns=timestamp_ns,
            width=width,
            height=height,
            channels=channels,
            format_type=format_type,
            mode=mode,
            image=image,
        )

    @staticmethod
    def _require_group(h5: h5py.File, group_name: str) -> h5py.Group:
        if group_name not in h5:
            available = ", ".join(h5.keys())
            raise KeyError(f"Group '{group_name}' not found. Available groups: {available}")
        group = h5[group_name]
        if not isinstance(group, h5py.Group):
            raise TypeError(f"'{group_name}' exists but is not an HDF5 group")
        return group

    @classmethod
    def _require_camera_group(cls, group: h5py.Group, group_name: str) -> None:
        missing = [name for name in cls.REQUIRED_DATASETS if name not in group]
        if missing:
            raise KeyError(f"Group '{group_name}' is not a camera image group; missing: {', '.join(missing)}")

    @staticmethod
    def _decode_image(
        payload: np.ndarray,
        width: int,
        height: int,
        channels: int,
        format_type: int,
        key: str,
    ) -> Tuple[np.ndarray, str]:
        if width <= 0 or height <= 0 or payload.size == 0:
            raise ValueError(f"Invalid frame metadata: width={width}, height={height}, bytes={payload.size}")

        key_l = key.lower()
        pixel_count = width * height

        # RealSense Z16 depth: RS2_FORMAT_Z16 == 1 and 2 bytes per pixel.
        if (format_type == 1 and channels == 2) or ("depth" in key_l and channels == 2):
            depth = payload.view(np.uint16)
            if depth.size < pixel_count:
                raise ValueError(f"Depth frame too small: got {depth.size}, expected {pixel_count} pixels")
            return depth[:pixel_count].reshape((height, width)), "depth"

        # Kinect depth/IR are libfreenect2 float32 frames with bytes_per_pixel == 4.
        if channels == 4 and (format_type in (2, 4) or "depth" in key_l):
            values = payload.view(np.float32)
            if values.size >= pixel_count:
                mode = "depth" if format_type == 4 or "depth" in key_l else "ir"
                return values[:pixel_count].reshape((height, width)), mode

        # RealSense RGB8: RS2_FORMAT_RGB8 == 5. Kinect color is commonly BGRX with type == 1.
        if format_type == 5 or "rgb" in key_l or (format_type == 1 and channels in (3, 4)):
            if channels <= 0:
                raise ValueError(f"Invalid channel count for RGB frame: {channels}")
            expected = pixel_count * channels
            if payload.size < expected:
                raise ValueError(f"RGB frame too small: got {payload.size}, expected {expected} bytes")
            rgb_like = payload[:expected].reshape((height, width, channels))
            if channels == 4:
                # libfreenect2 color data is usually BGRX; keep loader output in RGB order.
                return rgb_like[:, :, 2::-1], "rgb"
            if channels >= 3:
                return rgb_like[:, :, :3], "rgb"
            return rgb_like[:, :, 0], "gray"

        # RealSense IR Y8: RS2_FORMAT_Y8 == 9.
        if format_type == 9 and channels == 1:
            gray = payload[:pixel_count].reshape((height, width))
            return gray, "ir"

        if channels == 1 and payload.size >= pixel_count:
            return payload[:pixel_count].reshape((height, width)), "gray"

        raise ValueError(
            "Unsupported image encoding: "
            f"format_type={format_type}, channels={channels}, size={payload.size}, key={key}"
        )


def depth_to_colormap(image: np.ndarray) -> np.ndarray:
    import cv2

    values = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = values[values > 0]
    if valid.size:
        lo = float(np.percentile(valid, 1))
        hi = float(np.percentile(valid, 99))
    else:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((values - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)


def frame_to_bgr(frame: CameraFrame) -> np.ndarray:
    import cv2

    if frame.mode == "rgb":
        return cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
    if frame.mode in ("depth", "ir"):
        return depth_to_colormap(frame.image)
    if frame.image.ndim == 2:
        return cv2.cvtColor(frame.image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return frame.image


def print_groups(loader: H5CameraDataLoader) -> None:
    for group_name, datasets in loader.list_groups().items():
        print(f"{group_name}/")
        if not datasets:
            print("  <empty>")
            continue
        for name, (shape, dtype) in sorted(datasets.items()):
            print(f"  {name}: shape={shape} dtype={dtype}")


def print_frame_summary(label: str, frame: CameraFrame) -> None:
    print(
        f"{label}: key={frame.key} group={frame.group} index={frame.index} "
        f"mode={frame.mode} shape={frame.image.shape} dtype={frame.image.dtype} "
        f"width={frame.width} height={frame.height} channels={frame.channels} "
        f"format_type={frame.format_type} timestamp_ns={frame.timestamp_ns}"
    )


def _annotate(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    import cv2

    annotated = frame_bgr.copy()
    cv2.putText(
        annotated,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return annotated


def show_frames(rgb_frame: CameraFrame, depth_frame: CameraFrame) -> int:
    import cv2

    rgb_bgr = frame_to_bgr(rgb_frame)
    depth_bgr = frame_to_bgr(depth_frame)
    cv2.imshow("H5 RGB", _annotate(rgb_bgr, f"{rgb_frame.key} frame={rgb_frame.index}"))
    cv2.imshow("H5 Depth", _annotate(depth_bgr, f"{depth_frame.key} frame={depth_frame.index}"))
    return int(cv2.waitKey(1) & 0xFF)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read recorder_plugin HDF5 camera frames.")
    parser.add_argument("--h5", default=DEFAULT_H5, help=f"HDF5 file path (default: {DEFAULT_H5})")
    parser.add_argument("--rgb-key", default="realsense/rgb", help="RGB key or group name.")
    parser.add_argument("--depth-key", default="realsense/depth", help="Depth key or group name.")
    parser.add_argument("--index", type=int, default=0, help="Frame index to read.")
    parser.add_argument("--list", action="store_true", help="List HDF5 groups and datasets.")
    parser.add_argument("--play", action="store_true", help="Play RGB/depth frames from --index onward.")
    parser.add_argument("--fps", type=float, default=15.0, help="Playback FPS when --play is set.")
    parser.add_argument("--no-gui", action="store_true", help="Print frame stats without opening OpenCV windows.")
    args = parser.parse_args()

    loader = H5CameraDataLoader(args.h5)

    if args.list:
        print_groups(loader)
        if args.no_gui and not args.play:
            return 0

    if args.play and args.no_gui:
        print("--play requires GUI output; ignoring --play because --no-gui was provided.")
        args.play = False

    if not args.play:
        rgb_frame, depth_frame = loader.read_pair(args.rgb_key, args.depth_key, args.index)
        print_frame_summary("rgb", rgb_frame)
        print_frame_summary("depth", depth_frame)
        if args.no_gui:
            return 0

        show_frames(rgb_frame, depth_frame)
        print("Press any key in an OpenCV window to exit.")
        import cv2

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    import cv2

    rgb_count = loader.frame_count(args.rgb_key)
    depth_count = loader.frame_count(args.depth_key)
    end = min(rgb_count, depth_count)
    delay_s = 1.0 / max(args.fps, 1e-6)
    print(f"Playing {args.rgb_key} + {args.depth_key}: frames {args.index}..{end - 1}")

    index = max(0, args.index)
    while index < end:
        start = time.monotonic()
        rgb_frame, depth_frame = loader.read_pair(args.rgb_key, args.depth_key, index)
        key = show_frames(rgb_frame, depth_frame)
        if key in (27, ord("q")):
            break
        index += 1
        elapsed = time.monotonic() - start
        if elapsed < delay_s:
            time.sleep(delay_s - elapsed)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
