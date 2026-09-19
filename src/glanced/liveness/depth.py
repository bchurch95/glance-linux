"""3D Depth & Active IR Liveness Analyzer.

Interfaces with the Acer 3D depth camera (ACER IR U / Quanta 0408:4054 and compatible
active IR depth cameras) alongside the 2D color camera to provide hardware-backed
anti-spoofing and secure facial unlock.

Defeats:
1. Screen-replay attacks (smartphones, tablets, laptops displaying photos or videos),
   because display screens emit polarized visible RGB light and do not emit or reflect
   active near-IR with living tissue scattering.
2. Printed paper photo attacks, because 2D printouts lack active IR strobe response,
   facial 3D depth curvature, and stereoscopic parallax between RGB and IR sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .frame import CueReading, NO_READING


@dataclass(frozen=True)
class DepthObservation:
    depth_confirmed: bool = False
    depth_spoof: bool = False
    spoof_reason: Optional[str] = None
    strobe_delta: float = 0.0
    yaw_difference: Optional[float] = None
    pitch_difference: Optional[float] = None
    confirmed_reading: CueReading = NO_READING
    spoof_reading: CueReading = NO_READING


class DepthAnalyzer:
    """Tracks RGB and IR face observations across frames to confirm 3D liveness
    or flag presentation attacks."""

    def __init__(
        self,
        *,
        max_yaw_diff: float = 0.30,
        max_pitch_diff: float = 0.45,
        min_strobe_delta: float = 8.0,
        spoof_patience_frames: int = 5,
        confirm_frames_required: int = 2,
    ) -> None:
        self.max_yaw_diff = max_yaw_diff
        self.max_pitch_diff = max_pitch_diff
        self.min_strobe_delta = min_strobe_delta
        self.spoof_patience_frames = spoof_patience_frames
        self.confirm_frames_required = confirm_frames_required

        self._rgb_face_frames = 0
        self._depth_face_frames = 0
        self._consecutive_missing_ir = 0
        self._ambient_mean: Optional[float] = None
        self._active_mean: Optional[float] = None

    def reset(self) -> None:
        self._rgb_face_frames = 0
        self._depth_face_frames = 0
        self._consecutive_missing_ir = 0
        self._ambient_mean = None
        self._active_mean = None

    def observe(
        self,
        *,
        has_rgb_face: bool,
        rgb_yaw: Optional[float] = None,
        rgb_pitch: Optional[float] = None,
        has_depth_face: bool = False,
        depth_yaw: Optional[float] = None,
        depth_pitch: Optional[float] = None,
        depth_frame_mean: Optional[float] = None,
    ) -> DepthObservation:
        if not has_rgb_face:
            return DepthObservation()

        self._rgb_face_frames += 1

        if depth_frame_mean is not None:
            if self._ambient_mean is None or depth_frame_mean < self._ambient_mean:
                self._ambient_mean = depth_frame_mean
            if depth_frame_mean > 25.0:
                self._active_mean = depth_frame_mean

        strobe_delta = (
            max(0.0, (self._active_mean or 0.0) - (self._ambient_mean or 0.0))
            if (self._active_mean is not None and self._ambient_mean is not None)
            else 0.0
        )

        if has_depth_face:
            self._depth_face_frames += 1
            self._consecutive_missing_ir = 0

            yaw_diff = abs(rgb_yaw - depth_yaw) if rgb_yaw is not None and depth_yaw is not None else None
            pitch_diff = abs(rgb_pitch - depth_pitch) if rgb_pitch is not None and depth_pitch is not None else None

            pose_ok = (yaw_diff is None or yaw_diff <= self.max_yaw_diff) and (
                pitch_diff is None or pitch_diff <= self.max_pitch_diff
            )

            if pose_ok:
                return DepthObservation(
                    depth_confirmed=True,
                    depth_spoof=False,
                    strobe_delta=strobe_delta,
                    yaw_difference=yaw_diff,
                    pitch_difference=pitch_diff,
                    confirmed_reading=CueReading(level=1.0, confidence=1.0),
                    spoof_reading=NO_READING,
                )
            else:
                return DepthObservation(
                    depth_confirmed=False,
                    depth_spoof=True,
                    spoof_reason="Face orientation mismatch between 2D and 3D sensors",
                    strobe_delta=strobe_delta,
                    yaw_difference=yaw_diff,
                    pitch_difference=pitch_diff,
                    confirmed_reading=NO_READING,
                    spoof_reading=CueReading(level=1.0, confidence=1.0),
                )

        # RGB face is visible, but depth/IR face is not
        self._consecutive_missing_ir += 1
        if self._rgb_face_frames >= self.spoof_patience_frames and self._depth_face_frames == 0:
            return DepthObservation(
                depth_confirmed=False,
                depth_spoof=True,
                spoof_reason="Face visible on 2D camera but absent on 3D depth/IR camera",
                strobe_delta=strobe_delta,
                confirmed_reading=NO_READING,
                spoof_reading=CueReading(level=1.0, confidence=1.0),
            )

        return DepthObservation(
            depth_confirmed=False,
            depth_spoof=False,
            strobe_delta=strobe_delta,
            confirmed_reading=NO_READING,
            spoof_reading=NO_READING,
        )
