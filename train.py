# ============================================================================
# IMPORTS
# ============================================================================
import time
import wandb
import gymnasium as gym
from wandb.integration.sb3 import WandbCallback
from pathlib import Path
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

# ============================================================================
# CONFIGURATION
# ============================================================================
# Model and training hyperparameters
ALGORITHM = "A2C"
POLICY = "MlpPolicy"
TRAIN_TIMESTEPS = 20000
EVAL_FREQ = 500
MAX_EPISODE_STEPS = 500
INFERENCE_EPISODES = 1
SEED = 42
DEV = False  # Set to False to enable Weights & Biases logging

TIMESTRING = time.strftime("%Y%m%d-%H%M%S")

HYPERPARAMETERS = {
    "timesteps": TRAIN_TIMESTEPS,
    "max_steps_per_episode": MAX_EPISODE_STEPS,
    "algorithm": ALGORITHM,
    "policy": POLICY,
    "start_time": TIMESTRING,
}

MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

BEST_MODEL_DIR = MODELS_DIR / TIMESTRING / "best_model"
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

WANDB_DIR = MODELS_DIR / TIMESTRING / "wandb_checkpoints"
WANDB_DIR.mkdir(parents=True, exist_ok=True)

LOGS_OUTPUT_DIR = LOGS_DIR / TIMESTRING

TENSORBOARD_LOG_DIR = LOGS_OUTPUT_DIR / "tensorboard"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
# Create three environments: training (no render), evaluation (no render),
# and rendering (human mode for visualization during inference)
train_env = gym.make(
    "CartPole-v1", render_mode=None, max_episode_steps=MAX_EPISODE_STEPS
)
eval_env = gym.make(
    "CartPole-v1", render_mode=None, max_episode_steps=MAX_EPISODE_STEPS
)
render_env = gym.make(
    "CartPole-v1", render_mode="human", max_episode_steps=MAX_EPISODE_STEPS
)

# ============================================================================
# WANDB INITIALIZATION
# ============================================================================
if not DEV:
    run = wandb.init(
        project="CartPole-RL",
        name=f"{ALGORITHM}_{TIMESTRING}",
        config=HYPERPARAMETERS,
        sync_tensorboard=True,  # Automatically sync SB3 tensorboard logs
        monitor_gym=True,  # Automatically upload gym videos if available
        save_code=True,
    )

# ============================================================================
# CALLBACKS SETUP
# ============================================================================
# Configure evaluation callback to monitor training performance
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(BEST_MODEL_DIR),
    log_path=str(LOGS_OUTPUT_DIR),
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
)

callbacks = [eval_callback]
if not DEV:
    callbacks.append(
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=WANDB_DIR,
            verbose=2,
        )
    )

# Combine all callbacks
callback_list = CallbackList(callbacks)

# ============================================================================
# MODEL & TRAINING
# ============================================================================
# Initialize model with common parameters and train on collected experience
common_params = {
    "policy": POLICY,
    "env": train_env,
    "verbose": 1,
    "tensorboard_log": str(TENSORBOARD_LOG_DIR),
}

match ALGORITHM:
    case "PPO":
        model = PPO(**common_params)
    case "A2C":
        model = A2C(**common_params)

# Train the model
model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=callback_list, progress_bar=True)

if not DEV:
    run.finish()

train_env.close()

# ============================================================================
# INFERENCE
# ============================================================================
# Run trained model on rendering environment to visualize learned behavior
for episode in range(INFERENCE_EPISODES):
    observation, info = render_env.reset(seed=SEED)

    done = False
    while not done:
        # Get action from trained policy
        action, _states = model.predict(observation, deterministic=True)

        # Execute action and collect environment feedback
        observation, reward, terminated, truncated, info = render_env.step(action)

        # Render the environment state
        render_env.render()

        # Check if episode has ended
        done = terminated or truncated

eval_env.close()
render_env.close()
