"""
PPO training with 3-phase curriculum learning.

Phase 1 (empty):   Agent learns to navigate to the carrot with no obstacles.
Phase 2 (static):  Static boxes are scattered — agent learns avoidance.
Phase 3 (dynamic): Moving pedestrians are added — agent handles dynamic threats.

Each phase inherits the previous phase's weights, so knowledge carries over.

Run management:
  - Each training run gets its own auto-incremented folder: models/run_001/, run_002/, ...
  - Checkpoints are saved every 10k steps inside a 'checkpoints/' subfolder.
  - After all three phases complete successfully, checkpoints are deleted.
    Only the per-phase final models and the overall final model are kept.
"""

import os
import glob
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from rl_env import TurtleBotEnv

BASE_MODEL_DIR = "models"

PHASES = [
    {"name": "empty",   "phase": 1, "steps": 150_000},
    {"name": "static",  "phase": 2, "steps": 175_000},
    {"name": "dynamic", "phase": 3, "steps": 175_000},
]


def _next_run_dir(base):
    """Auto-increment: find the highest existing run_XXX folder and go one higher."""
    os.makedirs(base, exist_ok=True)
    existing = [
        int(d[4:]) for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")
    ]
    run_num = (max(existing) + 1) if existing else 1
    return os.path.join(base, f"run_{run_num:03d}")


def _purge_checkpoints(checkpoint_dir):
    """Delete the checkpoints folder once training completes cleanly."""
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"  Checkpoints removed: {checkpoint_dir}")


# ── Setup ──────────────────────────────────────────────────────────────────

RUN_DIR = _next_run_dir(BASE_MODEL_DIR)
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\nTraining run directory: {RUN_DIR}")
print(f"Checkpoints:           {CHECKPOINT_DIR}\n")

# Single env instance — we just swap the curriculum phase between rounds.
# Webots only allows one Supervisor per controller process, so we reuse it.
env = TurtleBotEnv(curriculum_phase=1)
model = PPO("MlpPolicy", env, verbose=1)

training_succeeded = False
try:
    for phase_info in PHASES:
        phase_name = phase_info["name"]
        phase_num  = phase_info["phase"]
        phase_steps = phase_info["steps"]

        print(f"\n{'='*60}")
        print(f"  CURRICULUM PHASE {phase_num}: {phase_name.upper()}")
        print(f"  Training for {phase_steps:,} timesteps")
        print(f"{'='*60}\n")

        env.set_curriculum_phase(phase_num)

        checkpoint_cb = CheckpointCallback(
            save_freq=10_000,
            save_path=CHECKPOINT_DIR,
            name_prefix=f"ckpt_phase{phase_num}_{phase_name}",
        )

        # reset_num_timesteps=False keeps the global step counter running
        # across phases — one continuous curve in TensorBoard, not three.
        model.learn(
            total_timesteps=phase_steps,
            callback=checkpoint_cb,
            reset_num_timesteps=False,
        )

        phase_model_path = os.path.join(RUN_DIR, f"ppo_phase{phase_num}_{phase_name}_final")
        model.save(phase_model_path)
        print(f"  Phase {phase_num} complete — saved: {phase_model_path}.zip")

    final_path = os.path.join(RUN_DIR, "ppo_final")
    model.save(final_path)
    print(f"\nAll phases complete.\nFinal model: {final_path}.zip")

    training_succeeded = True

finally:
    # Clean up checkpoints only when all phases finished cleanly.
    # If training crashed mid-way, checkpoints stay so you can inspect them.
    if training_succeeded:
        _purge_checkpoints(CHECKPOINT_DIR)
        print(f"\nRun complete. Models kept in: {RUN_DIR}")
    else:
        print(f"\nTraining interrupted — checkpoints preserved in: {CHECKPOINT_DIR}")
