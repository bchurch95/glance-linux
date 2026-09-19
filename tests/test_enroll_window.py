"""The enrollment ring's geometry, without opening a window.

Qt is an optional extra, so the whole module skips when it is absent rather
than failing a test run on a machine that only ever wanted the daemon.
"""

from __future__ import annotations

import pytest

from glanced import poses

pytest.importorskip("PySide6", reason="the enrollment window needs the 'gui' extra")

from glanced.gui import enroll_window  # noqa: E402 - after the skip


def test_ticks_divide_evenly_among_the_sectors():
    """A remainder would leave one direction with a short sector, and the
    mismatch would show as a gap in the ring rather than as an error."""
    assert enroll_window.TICK_COUNT % poses.SECTOR_COUNT == 0
    assert enroll_window.TICKS_PER_SECTOR == enroll_window.TICK_COUNT // poses.SECTOR_COUNT


def test_every_tick_belongs_to_exactly_one_sector():
    owners = [poses.sector_pose(i * 360.0 / enroll_window.TICK_COUNT)
              for i in range(enroll_window.TICK_COUNT)]
    assert all(owner is not None for owner in owners)
    counts = {pose.name: owners.count(pose) for pose in poses.SECTOR_POSES}
    assert set(counts.values()) == {enroll_window.TICKS_PER_SECTOR}


def test_the_completion_ring_sits_inside_the_lit_ticks():
    """It reads as a hair smaller once the ticks vanish, rather than landing
    exactly where their tips were."""
    tips = enroll_window.RING_DIAMETER / 2 + enroll_window.TICK_LENGTH_LIT
    outer = (enroll_window.RING_DIAMETER / 2 + enroll_window.TICK_LENGTH_LIT
             - enroll_window.COMPLETION_RING_INSET)
    assert outer < tips


def test_smooth_cropper_crops_and_smooths():
    import numpy as np
    from types import SimpleNamespace

    cropper = enroll_window.SmoothCropper(alpha=0.2)
    native = np.zeros((480, 640, 3), dtype=np.uint8)

    # First frame with face detected at (200, 100, 100, 100)
    obs1 = SimpleNamespace(native_bounding_box=(200.0, 100.0, 100.0, 100.0))
    img1 = cropper.crop(native, obs1)
    assert not img1.isNull()
    assert img1.width() == img1.height()
    first_cx = cropper.cx
    first_cy = cropper.cy

    # Second frame with face missing (e.g. head turn) - should NOT jump to center
    img2 = cropper.crop(native, None)
    assert not img2.isNull()
    assert cropper.cx == pytest.approx(first_cx)
    assert cropper.cy == pytest.approx(first_cy)

    # Face moves to (300, 200)
    obs3 = SimpleNamespace(native_bounding_box=(300.0, 200.0, 100.0, 100.0))
    cropper.crop(native, obs3)
    # EMA moved partially towards target
    assert first_cx < cropper.cx < 350.0

