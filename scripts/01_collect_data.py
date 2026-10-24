"""Step 1: Automated Expert Imitation Dataset Collection in MetaDrive."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import cv2
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env import create_driving_env
from src.env.expert_labeler import ExpertLabeler


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect automated expert imitation driving dataset (196x392 RGB + Telemetry + Laya Labels)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to collect (overrides config data.num_collect_frames).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Output dataset directory (overrides config data.data_dir).",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        default=None,
        choices=["metadrive", "carla"],
        help="Simulator backend ('metadrive' or 'carla').",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use lightweight kinematic/visual mock simulator without OpenGL.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_mock = bool(args.mock or cfg.get("project", {}).get("use_mock", False))
    simulator_name = (args.simulator or cfg.get("env", {}).get("simulator", "metadrive")).lower()
    num_frames = int(
        args.num_frames
        if args.num_frames is not None
        else cfg.get("data", {}).get("num_collect_frames", 2000)
    )
    data_dir = Path(
        args.data_dir
        if args.data_dir is not None
        else PROJECT_ROOT / cfg.get("data", {}).get("data_dir", "data")
    )
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "metadata.jsonl"

    env = create_driving_env(
        config=cfg,
        use_mock=use_mock,
        start_seed=int(cfg.get("env", {}).get("start_seed", 1000)),
        simulator=simulator_name,
    )
    labeler = ExpertLabeler()

    maneuver_counts: Counter[str] = Counter()
    speed_counts: Counter[int] = Counter()
    aeb_counts: Counter[int] = Counter()

    print(
        f"[01_collect_data] Collecting {num_frames} frames from '{simulator_name}' into '{data_dir}' (mock={use_mock})..."
    )

    collected = 0
    episode_idx = 0
    with open(meta_path, "w", encoding="utf-8") as meta_file, tqdm(
        total=num_frames, desc="Collecting Expert Frames", unit="frame"
    ) as pbar:
        while collected < num_frames:
            rgb, telemetry, info = env.reset(
                seed=int(cfg.get("env", {}).get("start_seed", 1000)) + episode_idx
            )
            episode_idx += 1
            done = False

            while not done and collected < num_frames:
                expert_action = env.get_expert_action(info)
                labels = labeler.label_step(telemetry, expert_action, info)

                rel_img_path = f"images/frame_{collected:06d}.png"
                abs_img_path = data_dir / rel_img_path
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(abs_img_path), bgr)

                record = {
                    "frame_id": collected,
                    "episode_id": episode_idx,
                    "image_path": rel_img_path,
                    "telemetry": telemetry,
                    "expert_action": [
                        round(float(expert_action[0]), 4),
                        round(float(expert_action[1]), 4),
                    ],
                    "labels": labels.to_dict(),
                }
                meta_file.write(json.dumps(record) + "\n")

                maneuver_counts[labels.maneuver] += 1
                speed_counts[labels.speed_level] += 1
                aeb_counts[labels.emergency_brake] += 1

                rgb, telemetry, _, done, info = env.step(expert_action)
                collected += 1
                pbar.update(1)

    env.close()

    print("\n" + "=" * 68)
    print(f"Dataset Collection Complete: {collected} frames saved to {data_dir}")
    print(f"  Maneuver distribution : {dict(maneuver_counts)}")
    print(f"  Speed level counts    : {dict(sorted(speed_counts.items()))}")
    print(f"  Emergency brake (AEB) : {dict(sorted(aeb_counts.items()))}")
    print("=" * 68)


if __name__ == "__main__":
    main()
