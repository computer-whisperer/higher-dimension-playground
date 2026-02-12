#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


NAME_RE = re.compile(
    r"^(?P<scene>.+)_(?P<shot>[^_]+(?:_[^_]+)*)_L(?P<layers>\d+)_S(?P<steps>\d+)\.png$"
)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def banding_score(img: np.ndarray) -> float:
    # Estimate sheet-like banding by looking for medium-frequency energy
    # along X in a central strip where the flat-horizon artifact is visible.
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    h, w = lum.shape
    y0 = int(h * 0.45)
    y1 = int(h * 0.70)
    x0 = int(w * 0.15)
    x1 = int(w * 0.85)
    strip = lum[y0:y1, x0:x1]
    if strip.size == 0:
        return 0.0
    centered = strip - strip.mean(axis=1, keepdims=True)
    freq = np.fft.rfft(centered, axis=1)
    power = (np.abs(freq) ** 2).mean(axis=0)
    if power.size < 8:
        return float(power.mean())
    lo = max(2, power.size // 32)
    hi = max(lo + 1, power.size // 5)
    return float(power[lo:hi].mean())


def parse_entry(path: Path):
    m = NAME_RE.match(path.name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "scene": d["scene"],
        "shot": d["shot"],
        "layers": int(d["layers"]),
        "steps": int(d["steps"]),
        "path": path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="frames/vte_quality_sweep",
        help="Directory containing sweep images.",
    )
    parser.add_argument(
        "--reference",
        default="frames/cpu_render.png",
        help="Reference image for RMSE (must match dimensions).",
    )
    parser.add_argument(
        "--filter",
        default="flat_flat_horizon",
        help="Only include files whose names contain this substring.",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.dir)
    ref_path = Path(args.reference)
    ref = load_rgb(ref_path) if ref_path.exists() else None

    entries = []
    for p in sorted(sweep_dir.glob("*.png")):
        if args.filter and args.filter not in p.name:
            continue
        parsed = parse_entry(p)
        if parsed is None:
            continue
        img = load_rgb(p)
        parsed["shape_ok"] = ref is not None and img.shape == ref.shape
        parsed["rmse"] = rmse(img, ref) if parsed["shape_ok"] else math.nan
        parsed["banding"] = banding_score(img)
        entries.append(parsed)

    if not entries:
        print("No matching images found.")
        return

    print("Per image:")
    print("scene,shot,L,S,rmse_to_ref,banding_score,file")
    for e in entries:
        rmse_str = f"{e['rmse']:.6f}" if not math.isnan(e["rmse"]) else "n/a"
        print(
            f"{e['scene']},{e['shot']},{e['layers']},{e['steps']},"
            f"{rmse_str},{e['banding']:.3f},{e['path'].name}"
        )

    grouped = defaultdict(list)
    for e in entries:
        grouped[(e["scene"], e["shot"], e["layers"])].append(e)

    print("\nCollapsed by (scene, shot, L):")
    print("scene,shot,L,rmse_mean,banding_mean,s_count")
    for (scene, shot, layers), g in sorted(grouped.items()):
        rmses = [x["rmse"] for x in g if not math.isnan(x["rmse"])]
        rmse_mean = float(np.mean(rmses)) if rmses else math.nan
        banding_mean = float(np.mean([x["banding"] for x in g]))
        rmse_str = f"{rmse_mean:.6f}" if not math.isnan(rmse_mean) else "n/a"
        print(f"{scene},{shot},{layers},{rmse_str},{banding_mean:.3f},{len(g)}")


if __name__ == "__main__":
    main()
