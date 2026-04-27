"""
PPO training with 3-phase curriculum.

Phase 1 (empty corridor): Navigate to carrot in the corridor (no obstacles).
Phase 2 (static boxes):   CardboardBoxes placed on the path.
Phase 3 (dynamic):        Moving pedestrians added.

Uses continuous action space and a corridor-based training world that
matches the evaluation world geometry.
"""

import os
import shutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from rl_env import TurtleBotEnv

BASE_MODEL_DIR = "models"

PHASES = [
    {"name": "empty",   "phase": 1, "steps": 200_000},
    {"name": "static",  "phase": 2, "steps": 400_000},
    # Phase 3 (dynamic) removed — causes catastrophic forgetting.
    # The agent learns to stand still and spin to avoid unpredictable pedestrians.
]

PLOT_EVERY_EPISODES = 200


class ConvergenceCallback(BaseCallback):
    """Save a convergence PNG every N episodes."""

    def __init__(self, run_dir, phases, plot_every=200):
        super().__init__(verbose=0)
        self.run_dir    = run_dir
        self.phases     = phases
        self.plot_every = plot_every
        self.plot_path  = os.path.join(run_dir, "convergence.png")

        self.ep_rewards = []
        self.ep_lengths = []
        self.phase_boundaries = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.ep_rewards.append(float(ep["r"]))
                self.ep_lengths.append(int(ep["l"]))

        if len(self.ep_rewards) > 0 and len(self.ep_rewards) % self.plot_every == 0:
            self._save_plot()
        return True

    def save_phase_snapshot(self, phase_num, phase_name):
        self.phase_boundaries.append(
            (self.num_timesteps, f"P{phase_num} end\n({phase_name})")
        )
        self._save_plot()

    def _save_plot(self):
        if len(self.ep_rewards) < 2:
            return

        rewards = np.array(self.ep_rewards)
        lengths = np.array(self.ep_lengths)
        episodes = np.arange(1, len(rewards) + 1)

        window = min(100, len(rewards))
        def rolling(arr, w):
            return np.convolve(arr, np.ones(w) / w, mode="valid")

        mean_r = rolling(rewards, window)
        mean_l = rolling(lengths, window)
        x_mean = episodes[window - 1:]

        fig = Figure(figsize=(14, 5))
        FigureCanvasAgg(fig)

        ax1 = fig.add_subplot(1, 2, 1)
        # Clip raw dots at ±500 so extreme outliers don't crush the y-axis.
        # Clip rolling mean separately at ±300 — this reveals the real trend
        # while still showing whether mean drifted outside the healthy range.
        plot_r = np.clip(rewards, -500, 500)
        mean_r_clip = np.clip(mean_r, -300, 200)
        ax1.plot(episodes, plot_r, alpha=0.2, color="#4C9BE8", linewidth=0.6)
        ax1.plot(x_mean, mean_r_clip, color="#1A5FAD", linewidth=2.0,
                 label=f"Rolling mean ({window} eps)")

        if len(rewards) > 0:
            total_steps = self.num_timesteps
            ep_per_step = len(rewards) / max(total_steps, 1)
            for step_ts, label in self.phase_boundaries:
                ep_at = int(step_ts * ep_per_step)
                if 0 < ep_at < len(rewards):
                    ax1.axvline(ep_at, color="red", linestyle="--",
                               linewidth=1.2, alpha=0.7)
                    ax1.text(ep_at + 2, ax1.get_ylim()[0], label,
                             fontsize=7, color="red", va="bottom")

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Episode Reward")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(episodes, lengths, alpha=0.2, color="#E8844C", linewidth=0.6)
        ax2.plot(x_mean, mean_l, color="#AD3E1A", linewidth=2.0,
                 label=f"Rolling mean ({window} eps)")

        if len(rewards) > 0:
            for step_ts, label in self.phase_boundaries:
                ep_at = int(step_ts * ep_per_step)
                if 0 < ep_at < len(lengths):
                    ax2.axvline(ep_at, color="red", linestyle="--",
                               linewidth=1.2, alpha=0.7)

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.set_title("Episode Length")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f"PPO Convergence — {len(rewards)} episodes  |  "
            f"Latest mean reward: {mean_r[-1]:.1f}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=130, bbox_inches="tight")
        print(f"  [plot] Updated → {self.plot_path}")


def _next_run_dir(base):
    os.makedirs(base, exist_ok=True)
    existing = [
        int(d[4:]) for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")
    ]
    run_num = (max(existing) + 1) if existing else 1
    return os.path.join(base, f"run_{run_num:03d}")


def _purge_checkpoints(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
        print(f"  Checkpoints removed: {d}")


# ── Setup ─────────────────────────────────────────────────────────────────────

RUN_DIR        = _next_run_dir(BASE_MODEL_DIR)
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
TB_LOG_DIR     = os.path.join(RUN_DIR, "tensorboard")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\nTraining run directory : {RUN_DIR}")
print(f"Checkpoints            : {CHECKPOINT_DIR}")
print(f"TensorBoard logs       : {TB_LOG_DIR}")
print(f"Convergence plot       : {RUN_DIR}/convergence.png")
print(f"\nTo watch live in TensorBoard run:")
print(f"  tensorboard --logdir={os.path.abspath(BASE_MODEL_DIR)}\n")

_raw_env = TurtleBotEnv(curriculum_phase=1)
env = Monitor(_raw_env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=TB_LOG_DIR,
    policy_kwargs={"net_arch": [128, 128]},
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

convergence_cb = ConvergenceCallback(
    run_dir=RUN_DIR,
    phases=PHASES,
    plot_every=PLOT_EVERY_EPISODES,
)

training_succeeded = False
try:
    for phase_info in PHASES:
        phase_name  = phase_info["name"]
        phase_num   = phase_info["phase"]
        phase_steps = phase_info["steps"]

        print(f"\n{'='*60}")
        print(f"  CURRICULUM PHASE {phase_num}: {phase_name.upper()}")
        print(f"  Training for {phase_steps:,} timesteps")
        print(f"{'='*60}\n")

        _raw_env.set_curriculum_phase(phase_num)

        checkpoint_cb = CheckpointCallback(
            save_freq=10_000,
            save_path=CHECKPOINT_DIR,
            name_prefix=f"ckpt_phase{phase_num}_{phase_name}",
        )

        model.learn(
            total_timesteps=phase_steps,
            callback=CallbackList([checkpoint_cb, convergence_cb]),
            reset_num_timesteps=False,
        )

        phase_path = os.path.join(RUN_DIR, f"ppo_phase{phase_num}_{phase_name}_final")
        model.save(phase_path)
        print(f"  Phase {phase_num} complete — saved: {phase_path}.zip")

        convergence_cb.save_phase_snapshot(phase_num, phase_name)

    final_path = os.path.join(RUN_DIR, "ppo_final")
    model.save(final_path)
    print(f"\nAll phases complete.\nFinal model: {final_path}.zip")
    training_succeeded = True

finally:
    convergence_cb._save_plot()
    if training_succeeded:
        _purge_checkpoints(CHECKPOINT_DIR)
        print(f"\nRun complete. Models in: {RUN_DIR}")
    else:
        print(f"\nTraining interrupted — checkpoints in: {CHECKPOINT_DIR}")
