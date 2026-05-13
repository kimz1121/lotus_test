"""Convert a LeRobot dataset (HF Hub) into a LIBERO/robomimic-style HDF5
that Lotus skill discovery (lotus/skill_learning/multisensory_repr/save_dinov2_repr.py)
can consume.

Required HDF5 keys per demo (what Lotus reads):
    data/demo_{i}/obs/agentview_rgb     (T, H, W, 3) uint8
    data/demo_{i}/obs/eye_in_hand_rgb   (T, H, W, 3) uint8
    data/demo_{i}/obs/joint_states      (T, 7)      float32
    data/demo_{i}/obs/gripper_states    (T, 2)      float32
    data/demo_{i}/obs/ee_states         (T, 6)      float32

We additionally write actions / dones / rewards / num_samples / states /
robot_states so downstream robomimic-style loaders also work. They are
placeholders where the LeRobot data does not provide an equivalent.

Mapping (RoboCasa LeRobot -> LIBERO). Image keys and state-vector layout are
auto-detected from ds.meta.info["features"], so both PandaOmron-style and
DAVIAN-Robotics/robocasa-H50-style datasets work:

    observation.images.robot0_agentview_left[_image]  -> obs/agentview_rgb (configurable)
    observation.images.robot0_eye_in_hand[_image]     -> obs/eye_in_hand_rgb

    observation.state[eef_pos]   (3)  -> obs/ee_states[0:3]
    observation.state[eef_quat]  (3)  -> obs/ee_states[3:6]   (xyz of xyzw quat; w dropped)
    observation.state[gripper_qpos] (2) -> obs/gripper_states
    joint_states                       -> zeros (LeRobot data has no joints)
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


AGENTVIEW_CHOICES = {
    "left": "observation.images.robot0_agentview_left",
    "right": "observation.images.robot0_agentview_right",
}
EYE_IN_HAND_KEY = "observation.images.robot0_eye_in_hand"


def _resolve_image_key(features: dict, base: str) -> str:
    """Find which variant of an image key actually exists in this dataset.

    DAVIAN-Robotics/robocasa-H50 names cameras with an `_image` suffix
    (`observation.images.robot0_agentview_left_image`); other RoboCasa LeRobot
    variants omit it. Try both.
    """
    for cand in (base, base + "_image"):
        if cand in features:
            return cand
    raise KeyError(
        f"Neither {base!r} nor {base + '_image'!r} found in dataset features. "
        f"Available image keys: {[k for k in features if k.startswith('observation.images.')]}"
    )


def _resolve_state_layout(features: dict) -> dict:
    """Locate eef_pos / eef_quat / gripper_qpos slices inside observation.state.

    Returns a dict of (start, stop) tuples keyed by 'eef_pos', 'eef_quat',
    'gripper_qpos'. Falls back to PandaOmron defaults if names are missing.
    """
    state = features.get("observation.state")
    if state is None:
        raise KeyError("observation.state not found in dataset features")
    names = state.get("names") or []
    name_to_idx = {n: i for i, n in enumerate(names)}

    def _find(prefix: str, n: int):
        idxs = [name_to_idx[f"{prefix}_{i}"] for i in range(n) if f"{prefix}_{i}" in name_to_idx]
        if len(idxs) != n:
            return None
        # Require a contiguous run.
        if max(idxs) - min(idxs) != n - 1:
            return None
        return (min(idxs), max(idxs) + 1)

    layout = {}
    # Try both common naming variants.
    for key, candidates, length in (
        ("eef_pos", ("robot0_base_to_eef_pos", "robot0_eef_pos"), 3),
        ("eef_quat", ("robot0_base_to_eef_quat", "robot0_eef_quat"), 4),
        ("gripper_qpos", ("robot0_gripper_qpos",), 2),
    ):
        for prefix in candidates:
            rng = _find(prefix, length)
            if rng is not None:
                layout[key] = rng
                break
    if not {"eef_pos", "eef_quat", "gripper_qpos"} <= set(layout):
        # PandaOmron-style fallback (the original hardcoded slicing).
        print(f"[warn] could not auto-detect state layout from names; "
              f"falling back to PandaOmron defaults. Names seen: {names}")
        layout = {
            "eef_pos": (7, 10),
            "eef_quat": (10, 14),
            "gripper_qpos": (14, 16),
        }
    return layout


def to_uint8_hwc(img_tensor: torch.Tensor, target_size: int | None) -> np.ndarray:
    """LeRobot images are float32 CHW in [0,1]. Return uint8 HWC, optionally resized."""
    arr = img_tensor.numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if target_size is not None and (arr.shape[0] != target_size or arr.shape[1] != target_size):
        import cv2
        arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return arr


def _episode_bounds(ds: LeRobotDataset, ep_idx: int) -> tuple[int, int]:
    ep = ds.meta.episodes[ep_idx]
    # Values may be list[int] (when loaded from disk) or int. Normalize.
    def _scalar(x):
        if isinstance(x, (list, tuple)):
            return int(x[0])
        return int(x)
    return _scalar(ep["dataset_from_index"]), _scalar(ep["dataset_to_index"])


def collect_episode(
    ds: LeRobotDataset,
    ep_idx: int,
    agentview_key: str,
    eye_in_hand_key: str,
    state_layout: dict,
    image_size: int | None,
):
    ep_from, ep_to = _episode_bounds(ds, ep_idx)
    T = ep_to - ep_from

    state_dim = ds.meta.info["features"]["observation.state"]["shape"][0]
    action_dim = ds.meta.info["features"]["action"]["shape"][0]

    agentview = np.empty((T, image_size or 256, image_size or 256, 3), dtype=np.uint8)
    eye_in_hand = np.empty_like(agentview)
    state = np.empty((T, state_dim), dtype=np.float32)
    action = np.empty((T, action_dim), dtype=np.float32)

    for t, frame_idx in enumerate(range(ep_from, ep_to)):
        sample = ds[frame_idx]
        agentview[t] = to_uint8_hwc(sample[agentview_key], image_size)
        eye_in_hand[t] = to_uint8_hwc(sample[eye_in_hand_key], image_size)
        state[t] = sample["observation.state"].numpy()
        action[t] = sample["action"].numpy()

    pos_a, pos_b = state_layout["eef_pos"]
    quat_a, _ = state_layout["eef_quat"]
    grip_a, grip_b = state_layout["gripper_qpos"]
    # Lotus expects ee_states (T, 6) → pos(3) + first 3 components of quat (drop w).
    ee_states = np.concatenate(
        [state[:, pos_a:pos_b], state[:, quat_a:quat_a + 3]], axis=1
    ).astype(np.float32)
    gripper_states = state[:, grip_a:grip_b].astype(np.float32)
    joint_states = np.zeros((T, 7), dtype=np.float32)

    return {
        "agentview_rgb": agentview,
        "eye_in_hand_rgb": eye_in_hand,
        "ee_states": ee_states,
        "gripper_states": gripper_states,
        "joint_states": joint_states,
        "state": state,
        "action": action,
    }


def write_demo(h5_data_grp: h5py.Group, demo_idx: int, ep: dict, task_name: str):
    T = ep["action"].shape[0]
    demo = h5_data_grp.create_group(f"demo_{demo_idx}")
    demo.attrs["num_samples"] = T
    demo.attrs["model_file"] = ""

    demo.create_dataset("actions", data=ep["action"], compression="gzip", compression_opts=4)
    demo.create_dataset("dones", data=np.zeros((T,), dtype=np.int64))
    demo.create_dataset("rewards", data=np.zeros((T,), dtype=np.float32))
    demo.create_dataset("states", data=ep["state"], compression="gzip", compression_opts=4)
    demo.create_dataset("robot_states", data=ep["state"], compression="gzip", compression_opts=4)

    obs = demo.create_group("obs")
    obs.create_dataset("agentview_rgb", data=ep["agentview_rgb"],
                       compression="gzip", compression_opts=4, chunks=(1, *ep["agentview_rgb"].shape[1:]))
    obs.create_dataset("eye_in_hand_rgb", data=ep["eye_in_hand_rgb"],
                       compression="gzip", compression_opts=4, chunks=(1, *ep["eye_in_hand_rgb"].shape[1:]))
    obs.create_dataset("joint_states", data=ep["joint_states"])
    obs.create_dataset("gripper_states", data=ep["gripper_states"])
    obs.create_dataset("ee_states", data=ep["ee_states"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True,
                   help="HuggingFace LeRobot dataset repo id, e.g. kimz1121/robocasa_...")
    p.add_argument("--output", required=True, help="Output .hdf5 path")
    p.add_argument("--agentview", choices=list(AGENTVIEW_CHOICES.keys()), default="left",
                   help="Which LeRobot agentview camera to map to LIBERO agentview_rgb")
    p.add_argument("--image-size", type=int, default=128,
                   help="Resize images to this square size (LIBERO default 128). Use 0 to keep original.")
    p.add_argument("--max-episodes", type=int, default=None, help="Optional cap on episode count")
    p.add_argument("--task-name", type=str, default="lerobot_task",
                   help="Stored as problem_info attribute on the data group")
    p.add_argument("--root", type=str, default=None,
                   help="Local LeRobot dataset root (skip HF download)")
    args = p.parse_args()

    image_size = args.image_size if args.image_size > 0 else None
    agentview_base = AGENTVIEW_CHOICES[args.agentview]

    print(f"Loading LeRobotDataset({args.repo_id}) ...")
    ds = LeRobotDataset(args.repo_id, root=args.root, download_videos=True)
    features = ds.meta.info["features"]
    agentview_key = _resolve_image_key(features, agentview_base)
    eye_in_hand_key = _resolve_image_key(features, EYE_IN_HAND_KEY)
    state_layout = _resolve_state_layout(features)

    n_ep = ds.num_episodes if args.max_episodes is None else min(ds.num_episodes, args.max_episodes)
    print(f"  num_episodes={ds.num_episodes}, using={n_ep}, fps={ds.fps}")
    print(f"  agentview={agentview_key} -> agentview_rgb")
    print(f"  {eye_in_hand_key} -> eye_in_hand_rgb")
    print(f"  state_layout={state_layout}")
    print(f"  image_size={image_size or 'native'}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        grp = f.create_group("data")
        env_args = {
            "env_name": args.task_name,
            "type": 1,
            "env_kwargs": {},
        }
        problem_info = {
            "language_instruction": args.task_name,
            "problem_name": args.task_name,
        }
        grp.attrs["env_args"] = json.dumps(env_args)
        grp.attrs["problem_info"] = json.dumps(problem_info)
        grp.attrs["total"] = 0

        total_frames = 0
        for ep_idx in range(n_ep):
            ep = collect_episode(ds, ep_idx, agentview_key, eye_in_hand_key, state_layout, image_size)
            write_demo(grp, ep_idx, ep, args.task_name)
            total_frames += ep["action"].shape[0]
            print(f"  demo_{ep_idx}: T={ep['action'].shape[0]}")

        grp.attrs["total"] = total_frames

    print(f"\nWrote {out_path}  ({n_ep} demos, {total_frames} frames)")


if __name__ == "__main__":
    main()
