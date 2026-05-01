#!/usr/bin/env python3

import cv2
import time
import json
import os
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path.home() / "ph6lite_cam"

CRAM0 = ROOT / "cram0"
FAST = ROOT / "cram_fast"
SLOW = ROOT / "cram_slow"
MRAMS = ROOT / "mram_s"
LOGS = ROOT / "logs"

for p in [CRAM0, FAST, SLOW, MRAMS, LOGS]:
    p.mkdir(parents=True, exist_ok=True)

DEVICE = "/dev/video1"

WIDTH = 640
HEIGHT = 480

FPS_START = 5
FPS_MAX = 10
FPS_RAMP_EVERY_FRAMES = 150
FPS_RAMP_STEP = 1
MAX_READ_FAIL_STREAK_FOR_RAMP = 0

FPS_TARGET = FPS_START

FAST_EVERY_N = 1
SLOW_EVERY_N = 10

MOTION_THRESHOLD = 18
MOTION_FRACTION_PASS = 0.02
LAPLACIAN_VAR_MIN = 20.0
BRIGHT_FRACTION_MAX = 0.30
DARK_FRACTION_MAX = 0.50
DARK_PIXEL_THRESHOLD = 35
BRIGHT_PIXEL_THRESHOLD = 220

prev_gray = None
prev_packet_hash = "GENESIS"


def blake2b256(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")

    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

    dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    os.fsync(dir_fd)
    os.close(dir_fd)


def normalize_frame(frame):
    resized = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def pseudo_measure(gray, prev_gray):
    brightness_min = int(np.min(gray))
    brightness_max = int(np.max(gray))
    brightness_avg = float(np.mean(gray))

    dark_fraction = float(np.mean(gray <= DARK_PIXEL_THRESHOLD))
    bright_fraction = float(np.mean(gray >= BRIGHT_PIXEL_THRESHOLD))

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))

    if prev_gray is None:
        motion_fraction = 0.0
        frame_delta_avg = 0.0
        frame_delta_max = 0
    else:
        delta = cv2.absdiff(gray, prev_gray)
        motion_mask = delta >= MOTION_THRESHOLD
        motion_fraction = float(np.mean(motion_mask))
        frame_delta_avg = float(np.mean(delta))
        frame_delta_max = int(np.max(delta))

    return {
        "brightness_min": brightness_min,
        "brightness_max": brightness_max,
        "brightness_avg": round(brightness_avg, 6),
        "dark_fraction": round(dark_fraction, 6),
        "bright_fraction": round(bright_fraction, 6),
        "laplacian_var": round(laplacian_var, 6),
        "entropy": round(entropy, 6),
        "motion_fraction": round(motion_fraction, 6),
        "frame_delta_avg": round(frame_delta_avg, 6),
        "frame_delta_max": frame_delta_max,
    }


def pseudo_adjudicate(measurement):
    if measurement["motion_fraction"] < MOTION_FRACTION_PASS:
        return {"verdict": "DROP", "reason": "NO_SIGNIFICANT_MOTION"}
    if measurement["laplacian_var"] < LAPLACIAN_VAR_MIN:
        return {"verdict": "DROP", "reason": "BLURRY"}
    if measurement["bright_fraction"] > BRIGHT_FRACTION_MAX:
        return {"verdict": "DROP", "reason": "OVEREXPOSED"}
    if measurement["dark_fraction"] > DARK_FRACTION_MAX:
        return {"verdict": "DROP", "reason": "UNDEREXPOSED"}
    return {"verdict": "PASS", "reason": "MOTION_TRIGGER"}


def soso_advisory(frame_id, measurement, previous_measurement):
    if previous_measurement is None:
        return {
            "schema": "ph6.soso_lite.v0.1",
            "authority": "ADVISORY_ONLY",
            "frame_id": frame_id,
            "state": "INIT",
            "brightness_delta_avg": 0.0,
            "dark_fraction_delta": 0.0,
            "bright_fraction_delta": 0.0,
            "motion_delta": 0.0,
            "note": "First frame; no previous frame comparison.",
        }

    brightness_delta_avg = measurement["brightness_avg"] - previous_measurement["brightness_avg"]
    dark_delta = measurement["dark_fraction"] - previous_measurement["dark_fraction"]
    bright_delta = measurement["bright_fraction"] - previous_measurement["bright_fraction"]
    motion_delta = measurement["motion_fraction"] - previous_measurement["motion_fraction"]

    if abs(brightness_delta_avg) > 15:
        state = "LIGHT_SHIFT"
    elif measurement["motion_fraction"] > 0.05:
        state = "ACTIVE_MOTION"
    elif measurement["frame_delta_avg"] > 8:
        state = "FRAME_CHANGE"
    else:
        state = "QUIET"

    return {
        "schema": "ph6.soso_lite.v0.1",
        "authority": "ADVISORY_ONLY",
        "frame_id": frame_id,
        "state": state,
        "brightness_delta_avg": round(brightness_delta_avg, 6),
        "dark_fraction_delta": round(dark_delta, 6),
        "bright_fraction_delta": round(bright_delta, 6),
        "motion_delta": round(motion_delta, 6),
        "note": "SoSo advisory packet. Not evidence authority.",
    }


def make_dashboard():
    return {
        "cam": "UNKNOWN",
        "fps": 0.0,
        "fps_target": FPS_START,
        "fast": "ACTIVE",
        "slow": f"EVERY {SLOW_EVERY_N}",
        "soso": "INIT",
        "pass_count": 0,
        "drop_count": 0,
        "frame_id": 0,
        "verdict": "NONE",
        "reason": "NONE",
        "motion_fraction": 0.0,
        "brightness_min": 0,
        "brightness_max": 0,
        "brightness_avg": 0.0,
        "laplacian_var": 0.0,
        "entropy": 0.0,
        "dark_fraction": 0.0,
        "bright_fraction": 0.0,
        "packet_hash_short": "NONE",
    }


def print_dashboard(d):
    os.system("clear")
    print("PH6-LITE CAMERA DASHBOARD")
    print("=" * 42)
    print(f"CAM:          {d['cam']}")
    print(f"FPS:          {d['fps']:.2f}")
    print(f"FPS TARGET:   {d['fps_target']}")
    print(f"FRAME:        {d['frame_id']}")
    print(f"FAST:         {d['fast']}")
    print(f"SLOW:         {d['slow']}")
    print(f"SoSo:         {d['soso']}")
    print("-" * 42)
    print(f"VERDICT:      {d['verdict']}")
    print(f"REASON:       {d['reason']}")
    print(f"PASS:         {d['pass_count']}")
    print(f"DROP:         {d['drop_count']}")
    print("-" * 42)
    print(f"MOTION:       {d['motion_fraction']:.6f}")
    print(f"LAPLACIAN:    {d['laplacian_var']:.2f}")
    print(f"ENTROPY:      {d['entropy']:.4f}")
    print(f"DARK FRAC:    {d['dark_fraction']:.4f}")
    print(f"BRIGHT FRAC:  {d['bright_fraction']:.4f}")
    print(f"BRIGHT MIN:   {d['brightness_min']}")
    print(f"BRIGHT MAX:   {d['brightness_max']}")
    print(f"BRIGHT AVG:   {d['brightness_avg']:.3f}")
    print("-" * 42)
    print(f"HASH:         {d['packet_hash_short']}")
    print("=" * 42)
    print("Rule: PSEUDO decides. SoSo only observes.")


def main():
    global prev_gray, prev_packet_hash

    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device: {DEVICE}")

    frame_id = 0
    previous_measurement = None
    start_time = time.time()
    dashboard = make_dashboard()
    _read_fail_streak = 0
    current_fps_target = FPS_START
    last_ramp_frame = 0

    while True:
        ok, frame = cap.read()
        ts = time.time()

        if not ok or frame is None:
            _read_fail_streak += 1

            if _read_fail_streak >= 5:
                dashboard["cam"] = "FAIL"
                dashboard["verdict"] = "NO_FRAME"
                dashboard["reason"] = "CAMERA_READ_FAILED"
                dashboard["soso"] = "NO_UPDATE"
                print_dashboard(dashboard)

            time.sleep(0.1)
            continue

        _read_fail_streak = 0
        dashboard["cam"] = "OK"

        if (
            current_fps_target < FPS_MAX
            and frame_id - last_ramp_frame >= FPS_RAMP_EVERY_FRAMES
            and _read_fail_streak <= MAX_READ_FAIL_STREAK_FOR_RAMP
        ):
            current_fps_target += FPS_RAMP_STEP
            last_ramp_frame = frame_id
            cap.set(cv2.CAP_PROP_FPS, current_fps_target)

        frame_id += 1

        gray = normalize_frame(frame)
        raw_hash = blake2b256(gray.tobytes())

        measurement = pseudo_measure(gray, prev_gray)
        adjudication = pseudo_adjudicate(measurement)
        soso_packet = soso_advisory(frame_id, measurement, previous_measurement)

        packet = {
            "schema": "ph6.frame_packet_lite.v0.1",
            "frame_id": frame_id,
            "timestamp": ts,
            "source": DEVICE,
            "width": WIDTH,
            "height": HEIGHT,
            "frame_hash": raw_hash,
            "previous_packet_hash": prev_packet_hash,
            "measurement": measurement,
            "adjudication": adjudication,
        }

        packet_bytes = json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        packet_hash = blake2b256(packet_bytes)
        packet["packet_hash"] = packet_hash

        atomic_write_json(CRAM0 / f"frame_{frame_id:012d}.json", packet)

        if frame_id % FAST_EVERY_N == 0:
            atomic_write_json(FAST / f"fast_{frame_id:012d}.json", packet)

        if frame_id % SLOW_EVERY_N == 0:
            slow_packet = dict(packet)
            slow_packet["lane"] = "SLOW"
            slow_packet["note"] = "Deferred deterministic review placeholder."
            atomic_write_json(SLOW / f"slow_{frame_id:012d}.json", slow_packet)

        atomic_write_json(MRAMS / f"soso_{frame_id:012d}.json", soso_packet)

        if adjudication["verdict"] == "PASS":
            dashboard["pass_count"] += 1
        else:
            dashboard["drop_count"] += 1

        elapsed = time.time() - start_time

        dashboard["cam"] = "OK"
        dashboard["fps"] = frame_id / elapsed if elapsed > 0 else 0.0
        dashboard["frame_id"] = frame_id
        dashboard["soso"] = soso_packet["state"]
        dashboard["verdict"] = adjudication["verdict"]
        dashboard["reason"] = adjudication["reason"]
        dashboard["motion_fraction"] = measurement["motion_fraction"]
        dashboard["laplacian_var"] = measurement["laplacian_var"]
        dashboard["entropy"] = measurement["entropy"]
        dashboard["dark_fraction"] = measurement["dark_fraction"]
        dashboard["bright_fraction"] = measurement["bright_fraction"]
        dashboard["brightness_min"] = measurement["brightness_min"]
        dashboard["brightness_max"] = measurement["brightness_max"]
        dashboard["brightness_avg"] = measurement["brightness_avg"]
        dashboard["packet_hash_short"] = packet_hash[:16]
        dashboard["fps_target"] = current_fps_target

        print_dashboard(dashboard)

        previous_measurement = measurement
        prev_gray = gray
        prev_packet_hash = packet_hash

        time.sleep(max(0, (1.0 / current_fps_target)))


if __name__ == "__main__":
    main()
