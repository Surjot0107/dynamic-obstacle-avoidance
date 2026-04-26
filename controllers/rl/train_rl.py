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

Convergence monitoring (two methods):
  1. TensorBoard — pass  tensorboard --logdir=controllers/rl/models  in a terminal
     to get live ep_rew_mean / ep_len_mean / policy_loss curves.
  2. PNG plots — a convergence plot is saved to run_XXX/ after every phase
     and updated in-place throughout training.  No server needed.
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
    {"name": "static",  "phase": 2, "steps": 250_000},
    {"name": "dynamic", "phase": 3, "steps": 250_000},
]

PLOT_EVERY_EPISODES = 200   # re-draw the PNG every N completed episodes


# ── Convergence-plot callback ─────────────────────────────────────────────────

class ConvergenceCallback(BaseCallback):
    """
    Tracks per-episode rewards and lengths, then writes a PNG convergence
    plot to <run_dir>/convergence.png after every PLOT_EVERY_EPISODES episodes
    and at the end of each phase.

    Vertical phase-boundary lines are drawn so you can see exactly where
    curriculum transitions happened on the reward curve.

    SB3 stores completed-episode info in self.locals["infos"][i]["episode"]
    when a done flag fires.  We collect those values here instead of
    using a Monitor wrapper, which would require forking the env.
    """

    def __init__(self, run_dir: str, phases: list, plot_every: int = 50):
        super().__init__(verbose=0)
        self.run_dir     = run_dir
        self.phases      = phases          # list of phase dicts (name, steps)
        self.plot_every  = plot_every
        self.plot_path   = os.path.join(run_dir, "convergence.png")

        # Per-episode accumulators
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int]   = []

        # Timestep at which each phase boundary was crossed (filled in at runtime)
        self.phase_boundaries: list[tuple[int, str]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.ep_rewards.append(float(ep["r"]))
                self.ep_lengths.append(int(ep["l"]))

        n_eps = len(self.ep_rewards)
        if n_eps > 0 and n_eps % self.plot_every == 0:
            self._save_plot()

        return True   # returning False would stop training

    def save_phase_snapshot(self, phase_num: int, phase_name: str):
        """Call this at the end of each phase to record the boundary and save."""
        self.phase_boundaries.append((self.num_timesteps, f"P{phase_num} end\n({phase_name})"))
        self._save_plot()

    def _save_plot(self):
        if len(self.ep_rewards) < 2:
            return  # not enough data yet

        rewards = np.array(self.ep_rewards)
        lengths = np.array(self.ep_lengths)
        episodes = np.arange(1, len(rewards) + 1)

        # Rolling mean (window = min(100, available episodes))
        window = min(100, len(rewards))
        def rolling(arr, w):
            return np.convolve(arr, np.ones(w) / w, mode="valid")

        mean_r = rolling(rewards, window)
        mean_l = rolling(lengths, window)
        x_mean = episodes[window - 1:]

        fig = Figure(figsize=(14, 5))
        FigureCanvasAgg(fig)

        # ── Left panel: episode reward ────────────────────────────────────
        ax1 = fig.add_subplot(1, 2, 1)
        
        # Clip rewards for plotting so deep negative outliers don't squash the graph
        plot_rewards = np.clip(rewards, -500, 300)
        mean_r_clipped = np.clip(mean_r, -500, 300)
        
        ax1.plot(episodes, plot_rewards, alpha=0.25, color="#4C9BE8", linewidth=0.8, label="Episode reward (clipped)")
        ax1.plot(x_mean, mean_r_clipped, color="#1A5FAD", linewidth=2.0, label=f"Rolling mean ({window} eps)")

        # Phase boundary lines
        if len(rewards) > 0:
            total_steps = self.num_timesteps
            ep_per_step = len(rewards) / max(total_steps, 1)
            for step_ts, label in self.phase_boundaries:
                ep_at_boundary = int(step_ts * ep_per_step)
                if 0 < ep_at_boundary < len(rewards):
                    ax1.axvline(ep_at_boundary, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
                    ax1.text(ep_at_boundary + 2, ax1.get_ylim()[0], label,
                             fontsize=7, color="red", va="bottom")

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Episode Reward")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Right panel: episode length ───────────────────────────────────
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(episodes, lengths, alpha=0.25, color="#E8844C", linewidth=0.8, label="Episode length")
        ax2.plot(x_mean, mean_l, color="#AD3E1A", linewidth=2.0, label=f"Rolling mean ({window} eps)")

        for step_ts, label in self.phase_boundaries:
            ep_at_boundary = int(step_ts * ep_per_step)
            if 0 < ep_at_boundary < len(lengths):
                ax2.axvline(ep_at_boundary, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

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
        print(f"  [plot] Convergence chart updated → {self.plot_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

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

def linear_schedule(initial_lr):
    """Linearly decay LR from initial_lr to 0 over training."""
    def fn(progress_remaining):
        return progress_remaining * initial_lr
    return fn

# Single env — Webots only allows one Supervisor per controller process
_raw_env = TurtleBotEnv(curriculum_phase=1)
env = Monitor(_raw_env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=TB_LOG_DIR,

    # ── Network architecture ─────────────────────────────────────────────
    # Default is [64, 64]. Our obs is 76-dim, so we need more capacity.
    policy_kwargs={"net_arch": [256, 256]},

    # ── Learning rate ────────────────────────────────────────────────────
    # Linear decay: starts at 3e-4, reaches 0 at end of training.
    # Prevents overshooting the optimal policy in late training.
    learning_rate=linear_schedule(3e-4),

    # ── Rollout buffer ───────────────────────────────────────────────────
    # n_steps per update. Smaller than default (2048) → more frequent
    # gradient updates, which helps with Webots' slow wall-clock speed.
    n_steps=1024,
    batch_size=64,
    n_epochs=10,

    # ── PPO core ─────────────────────────────────────────────────────────
    gamma=0.99,        # discount — 0.99 = agent cares about ~100 steps ahead
    gae_lambda=0.95,   # GAE smoothing
    clip_range=0.2,    # PPO clip range — keep in [0.1, 0.3]

    # ── Entropy bonus ────────────────────────────────────────────────────
    # ent_coef > 0 forces the policy to stay spread across all 9 actions.
    # Without this, the policy collapses to ~3 actions and stops exploring.
    ent_coef=0.01,

    # ── Value function ───────────────────────────────────────────────────
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

        # reset_num_timesteps=False keeps the global step counter ticking
        # across phases — one continuous curve in TensorBoard, not three.
        model.learn(
            total_timesteps=phase_steps,
            callback=CallbackList([checkpoint_cb, convergence_cb]),
            reset_num_timesteps=False,
        )

        # Save the per-phase model and stamp the convergence plot with the
        # phase boundary before moving on to the next curriculum stage.
        phase_model_path = os.path.join(RUN_DIR, f"ppo_phase{phase_num}_{phase_name}_final")
        model.save(phase_model_path)
        print(f"  Phase {phase_num} complete — saved: {phase_model_path}.zip")

        convergence_cb.save_phase_snapshot(phase_num, phase_name)

    final_path = os.path.join(RUN_DIR, "ppo_final")
    model.save(final_path)
    print(f"\nAll phases complete.\nFinal model: {final_path}.zip")

    training_succeeded = True

finally:
    # Final plot including all three phase boundaries
    convergence_cb._save_plot()

    if training_succeeded:
        _purge_checkpoints(CHECKPOINT_DIR)
        print(f"\nRun complete. Models + convergence plot in: {RUN_DIR}")
    else:
        print(f"\nTraining interrupted — checkpoints preserved in: {CHECKPOINT_DIR}")
