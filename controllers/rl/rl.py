"""
Webots controller entry point for the RL agent.

Webots requires the main script to match the controller folder name,
so this thin dispatcher delegates to the actual training or evaluation
module. Flip MODE before launching the simulation.
"""

MODE = "evaluate"   # "train" or "evaluate"

if MODE == "train":
    import train_rl
elif MODE == "evaluate":
    import evaluate_rl
else:
    raise ValueError(f"Unknown MODE: {MODE!r}. Use 'train' or 'evaluate'.")
