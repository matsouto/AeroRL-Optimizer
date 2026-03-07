# TODO: Melhorar prints no terminal

# ============================================================================
# IMPORTS
# ============================================================================
import time
import wandb
import gymnasium as gym
from pathlib import Path
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from src.enviroments import StaticOptEnv, MultiStageOptEnv

# ============================================================================
# CONFIGURATION
# ============================================================================
# Model and training hyperparameters

ALGORITHM = "PPO"
POLICY = "MlpPolicy"
TRAIN_TIMESTEPS = 300000
MAX_EPISODE_STEPS = 50
LATENT_DIM = 16
ACTION_RANGE = 0.5
LATENT_RANGE = 2.0
EVAL_FREQ = 400
INFERENCE_EPISODES = 5
DEV = False  # Set to False to enable Weights & Biases logging

TIMESTRING = time.strftime("%Y%m%d-%H%M%S")

HYPERPARAMETERS = {
    "timesteps": TRAIN_TIMESTEPS,
    "max_steps_per_episode": MAX_EPISODE_STEPS,
    "algorithm": ALGORITHM,
    "policy": POLICY,
    "start_time": TIMESTRING,
    "latent_dim": LATENT_DIM,
    "latent_range": LATENT_RANGE,
    "action_range": ACTION_RANGE,
}

PROJECT_PATH = Path(__file__).resolve().parent
MODELS_PATH = PROJECT_PATH / "models"
ONNX_MODELS_PATH = MODELS_PATH / "onnx_decoder"

# 1. Convert iterdir() to a list to check its length safely
onnx_files = list(ONNX_MODELS_PATH.iterdir())

if len(onnx_files) != 2:
    raise ValueError(
        f"Expected exactly 2 files in {ONNX_MODELS_PATH} (Decoder and Scaler), but found {len(onnx_files)}."
    )

# 2 & 3. Convert glob generator to a list, extract the first item, and assign it directly.
# Since pathlib handles the absolute/relative paths internally, this is perfectly safe.
SCALER_PATH = list(ONNX_MODELS_PATH.glob("*.pkl"))[0]
DECODER_PATH = list(ONNX_MODELS_PATH.glob("*.onnx"))[0]

# Optional: Print them out to verify they look correct
print(f"Loaded Scaler from: {SCALER_PATH}")
print(f"Loaded Decoder from: {DECODER_PATH}")

LOGS_DIR = Path("logs")

BEST_MODEL_DIR = MODELS_PATH / TIMESTRING / "best_model"
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

WANDB_DIR = MODELS_PATH / TIMESTRING / "wandb_checkpoints"
WANDB_DIR.mkdir(parents=True, exist_ok=True)

LOGS_OUTPUT_DIR = LOGS_DIR / TIMESTRING

TENSORBOARD_LOG_DIR = LOGS_OUTPUT_DIR / "tensorboard"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
# Create three environments: training (no render), evaluation (no render),
# and rendering (human mode for visualization during inference)
def make_custom_env(render_mode=None):
    return MultiStageOptEnv(
        scaler_path=SCALER_PATH,
        decoder_path=DECODER_PATH,
        latent_dim=LATENT_DIM,
        action_range=ACTION_RANGE,
        latent_range=LATENT_RANGE,
        max_episode_steps=MAX_EPISODE_STEPS,
        n_alphas=40,
        lower_alpha=-5.0,
        upper_alpha=15.0,
        render_mode=render_mode,
    )


train_env = make_custom_env(render_mode=None)
eval_env = make_custom_env(render_mode=None)
render_env = make_custom_env(render_mode="human")

# ============================================================================
# WANDB INITIALIZATION
# ============================================================================
if not DEV:
    run = wandb.init(
        project="StaticOpt",
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

# Print training configuration
print("\n" + "=" * 70)
print("TRAINING STARTED")
print("=" * 70)
print(f"Algorithm: {ALGORITHM}")
print(f"Policy: {POLICY}")
print(f"Total Timesteps: {TRAIN_TIMESTEPS:,}")
print(f"Max Episode Steps: {MAX_EPISODE_STEPS}")
print(f"Latent Dimension: {LATENT_DIM}")
print(f"Evaluation Frequency: {EVAL_FREQ}")
print(f"Model Save Directory: {BEST_MODEL_DIR}")
print(f"Tensorboard Log Directory: {TENSORBOARD_LOG_DIR}")
if not DEV:
    print(f"Weights & Biases: ENABLED")
else:
    print(f"Weights & Biases: DISABLED (DEV MODE)")
print("=" * 70 + "\n")

# Train the model
model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=callback_list, progress_bar=True)

# Print training completion summary
print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)
print(f"Algorithm Used: {ALGORITHM}")
print(f"Total Timesteps: {TRAIN_TIMESTEPS:,}")
print(f"Best Model Saved: {BEST_MODEL_DIR / 'best_model.zip'}")
print(f"Evaluation Logs: {LOGS_OUTPUT_DIR}")
print(f"Tensorboard Logs: {TENSORBOARD_LOG_DIR}")
if not DEV:
    print(f"Weights & Biases Artifacts: {WANDB_DIR}")
    run.finish()
print("=" * 70 + "\n")

if not DEV:
    pass  # run.finish() already called above

train_env.close()

# Input so that user can see training results before inference starts
input("\n\nTraining complete! Press Enter to start inference and visualization...")

# ============================================================================
# INFERENCE
# ============================================================================
# Run trained model on rendering environment to visualize learned behavior
for episode in range(INFERENCE_EPISODES):
    # observation, info = render_env.reset(seed=SEED)
    observation, info = render_env.reset()

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
