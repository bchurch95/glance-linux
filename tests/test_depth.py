from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from glanced.camera import CameraConfig, FramePair, find_paired_depth_camera
from glanced.liveness import (
    CueRole,
    DecisionKind,
    LivenessAnalyzer,
    LivenessCue,
    LivenessFrame,
    LivenessMode,
)
from glanced.liveness.depth import DepthAnalyzer, DepthObservation
from glanced.liveness.frame import CueReading
from glanced.pipeline import Outcome, UnlockPipeline
from glanced.store import EnrollmentStore, Identity


def test_depth_cues_registered():
    assert LivenessCue.DEPTH_CONFIRMED in LivenessCue
    assert LivenessCue.DEPTH_SPOOF in LivenessCue
    assert LivenessCue.DEPTH_CONFIRMED.role is CueRole.CONFIRM
    assert LivenessCue.DEPTH_SPOOF.role is CueRole.DENY
    assert "depth" in LivenessCue.DEPTH_CONFIRMED.explanation.lower()
    assert "depth" in LivenessCue.DEPTH_SPOOF.explanation.lower()


def test_depth_analyzer_live_face():
    analyzer = DepthAnalyzer()
    obs = analyzer.observe(
        has_rgb_face=True,
        rgb_yaw=-0.12,
        rgb_pitch=0.05,
        has_depth_face=True,
        depth_yaw=-0.13,
        depth_pitch=0.06,
        depth_frame_mean=45.0,
    )
    assert obs.depth_confirmed is True
    assert obs.depth_spoof is False
    assert obs.confirmed_reading.level == 1.0
    assert obs.confirmed_reading.confidence == 1.0
    assert obs.spoof_reading.confidence == 0.0
    assert obs.yaw_difference is not None and obs.yaw_difference < 0.05


def test_depth_analyzer_pose_mismatch():
    analyzer = DepthAnalyzer(max_yaw_diff=0.30)
    obs = analyzer.observe(
        has_rgb_face=True,
        rgb_yaw=0.0,
        rgb_pitch=0.0,
        has_depth_face=True,
        depth_yaw=0.65,  # Mismatched pose
        depth_pitch=0.0,
        depth_frame_mean=40.0,
    )
    assert obs.depth_confirmed is False
    assert obs.depth_spoof is True
    assert obs.spoof_reading.level == 1.0
    assert "mismatch" in (obs.spoof_reason or "").lower()


def test_depth_analyzer_screen_spoof_detection():
    analyzer = DepthAnalyzer(spoof_patience_frames=4)
    # Simulate an attacker presenting a phone/screen: RGB face is visible, but IR sensor sees nothing
    for i in range(3):
        obs = analyzer.observe(has_rgb_face=True, has_depth_face=False)
        assert obs.depth_spoof is False  # within patience window

    # On the 4th frame without depth, spoof detection triggers
    obs = analyzer.observe(has_rgb_face=True, has_depth_face=False)
    assert obs.depth_spoof is True
    assert obs.depth_confirmed is False
    assert obs.spoof_reading.level == 1.0
    assert "absent" in (obs.spoof_reason or "").lower()


def test_pipeline_require_depth():
    store = MagicMock(spec=EnrollmentStore)
    id_test = Identity(name="alice", embeddings=[np.zeros(512, dtype=np.float32)])
    store.match.return_value = (id_test, 0.95)

    pipeline = UnlockPipeline(store, mode=LivenessMode.LIGHT, require_depth=True)
    pipeline.begin()

    # Frame 1: Match found and Light mode would auto-confirm, but depth is required and NOT confirmed yet
    frame_no_depth = LivenessFrame(timestamp=1.0)
    dummy_embedding = np.zeros(512, dtype=np.float32)

    # Observe 3 frames without depth
    for t in (1.0, 1.05, 1.10):
        frame = LivenessFrame(timestamp=t)
        res = pipeline.observe(frame, dummy_embedding)
        assert res.outcome is Outcome.PENDING  # Held pending because depth is not confirmed!

    # Now feed depth confirmed frames
    for t in (1.15, 1.20):
        frame = LivenessFrame(
            timestamp=t,
            depth_reading=CueReading(level=1.0, confidence=1.0),
        )
        res = pipeline.observe(frame, None)

    assert res.outcome is Outcome.UNLOCKED
    assert res.identity == id_test


def test_pipeline_depth_spoof_denied():
    store = MagicMock(spec=EnrollmentStore)
    id_test = Identity(name="alice", embeddings=[np.zeros(512, dtype=np.float32)])
    store.match.return_value = (id_test, 0.95)

    pipeline = UnlockPipeline(store, mode=LivenessMode.LIGHT, require_depth=True)
    pipeline.begin()

    dummy_embedding = np.zeros(512, dtype=np.float32)
    # Feed depth spoof frames (4 frames required by default tuning)
    res = Outcome.PENDING
    for i in range(5):
        frame = LivenessFrame(
            timestamp=1.0 + i * 0.05,
            depth_spoof_reading=CueReading(level=1.0, confidence=1.0),
        )
        res = pipeline.observe(frame, dummy_embedding if i == 0 else None)

    assert res.outcome is Outcome.SPOOF_DENIED
    assert "Depth check failed" in (res.reason or "")


def test_find_paired_depth_camera_hardware():
    # If on the machine with Acer FHD User Facing camera, verify it discovers /dev/video2
    res = find_paired_depth_camera("/dev/video0")
    if res is not None:
        assert res == "/dev/video2"
