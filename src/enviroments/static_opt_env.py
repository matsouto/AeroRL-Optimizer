import joblib
import gymnasium as gym
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path
from gymnasium import spaces
from aerosandbox import Airfoil
from scipy.interpolate import interp1d
from src.helpers import cst_to_coords

from src.airfoil import airfoil_modifications


class StaticOptEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        scaler_path: str,
        decoder_path: str,
        latent_dim: int = 16,
        action_range: float = 0.1,
        latent_range: float = 3.0,
        max_episode_steps: int = 50,
        n_alphas: int = 40,
        lower_alpha: float = -5.0,
        upper_alpha: float = 15.0,
        render_mode: str = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_range = action_range
        self.latent_range = latent_range
        self.n_alphas = n_alphas
        self.lower_alpha = lower_alpha
        self.upper_alpha = upper_alpha
        self.max_episode_steps = max_episode_steps
        self.scaler_path = scaler_path
        self.decoder_path = decoder_path
        self.render_mode = render_mode

        # Initialize latent vector and aerodynamic coefficients
        self._current_z = np.zeros(self.latent_dim, dtype=np.float32)
        self._current_cl_sweep = np.zeros(self.n_alphas, dtype=np.float32)
        self._current_cd_sweep = np.zeros(self.n_alphas, dtype=np.float32)
        self._current_efficiency = 0.0
        self._analysis_confidence = 0.0

        # Initialize step counter
        self._current_step = 0

        # Action space: adjustments to the latent vector
        self.action_space = spaces.Box(
            low=-self.action_range,
            high=self.action_range,
            shape=(self.latent_dim,),
            dtype=np.float32,
        )

        # Observation space: latent vector
        self.observation_space = spaces.Box(
            low=-self.latent_range,
            high=self.latent_range,
            shape=(self.latent_dim,),
            dtype=np.float32,
        )

        # Initialize scaler
        self.scaler = joblib.load(scaler_path)

        # Initialize ONNX runtime session for the decoder
        self.session = ort.InferenceSession(decoder_path)
        self.input_name = self.session.get_inputs()[0].name

    def _get_obs(self):
        # return np.concatenate([self._current_z, self._current_e])
        return self._current_z

    def _get_info(self):
        return {
            "cl_sweep": self._current_cl_sweep,
            "cd_sweep": self._current_cd_sweep,
            "efficiency": self._current_efficiency,
        }

    def _get_coords(self):
        # Run ONNX inference on decoder
        outputs = self.session.run(
            None, {self.input_name: self._current_z.reshape(1, -1)}
        )

        # ONNX returns normalized weight and pressure tensors
        w_norm = outputs[0]
        p_norm = outputs[1]

        # Denormalize using scaler
        w_phys, p_phys = self.scaler.inverse_transform(w_norm, p_norm)

        # Generate CST coordinates
        x_coords, y_coords = cst_to_coords(w_phys[0], p_phys[0], n_points=100)

        coords = np.stack((x_coords, y_coords), axis=-1)

        return coords

    def step(self, action):
        self._current_step += 1

        # Update latent vector with action
        self._current_z += action

        # Clip to stay within valid bounds
        self._current_z = np.clip(
            self._current_z, -self.latent_range, self.latent_range
        )

        # Get airfoil coordinates from decoder
        coords = self._get_coords()

        try:
            af = Airfoil(coordinates=coords)
            aero = af.get_aero_from_neuralfoil(
                alpha=np.linspace(0, 18, self.n_alphas),
                Re=1e6,
                mach=0.2,
                model_size="xlarge",
            )

            self._current_cl_sweep = aero["CL"]
            self._current_cd_sweep = aero["CD"]
            confidence = aero["analysis_confidence"]

            valid_mask = confidence >= 0.30

            if not np.any(valid_mask):
                reward = -1.0  # Fixed penalty for low confidence
                self._current_efficiency = 0.0
            else:
                valid_cl = self._current_cl_sweep[valid_mask]
                valid_cd = self._current_cd_sweep[valid_mask]

                # Protect against division by zero with minimum CD value
                safe_cd = np.maximum(valid_cd, 0.005)

                # Calculate efficiency with upper bound (even best airfoils rarely exceed 200)
                eff_sweep = valid_cl / safe_cd
                raw_eff = float(np.max(eff_sweep))
                self._current_efficiency = np.clip(raw_eff, 0, 250)

                # Calculate airfoil thickness
                coords = self._get_coords()
                max_thickness = af.max_thickness()

                thickness_penalty = 0.0

                # Penalty for airfoils too thin (minimum 10% thickness)
                if max_thickness < 0.10:
                    diff = 0.10 - max_thickness
                    # Squared penalty for smooth soft-wall
                    thickness_penalty = (diff**2) * 1000.0

                reward = (self._current_efficiency - thickness_penalty) / 100.0

        except Exception as e:
            reward = -2.0
            self._current_efficiency = 0.0

        terminated = False
        truncated = self._current_step >= self.max_episode_steps

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0

        # Sample a new random latent vector for the episode using gymnasium's managed RNG
        self._current_z = self.np_random.normal(0, 1, size=(self.latent_dim,)).astype(
            np.float32
        )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self):
        coords = self._get_coords()
        x_coords, y_coords = coords[:, 0], coords[:, 1]

        af = Airfoil(coordinates=coords)

        # Get airfoil characteristics
        max_thickness = af.max_thickness()
        camber = af.max_camber()
        le_radius = af.LE_radius()

        # Enable interactive mode
        plt.ion()

        # Create or reuse figure with horizontal aspect ratio
        if not plt.fignum_exists(1):
            plt.figure(1, figsize=(12, 5))

        # Clear previous frame
        plt.clf()

        # ==================================================
        # Subplot 1: Airfoil geometry
        # ==================================================
        plt.subplot(1, 2, 1)
        plt.plot(x_coords, y_coords, color="blue", linewidth=2)
        plt.fill(x_coords, y_coords, color="blue", alpha=0.15)

        plt.title(
            f"Airfoil | Thickness: {max_thickness:.2f}% | Camber: {camber:.3f}% | LE Radius: {le_radius:.4f}"
        )
        plt.xlabel("Chord (X)")
        plt.ylabel("Thickness (Y)")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Lock axes to prevent scaling
        plt.axis("equal")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.25, 0.25)

        # ==================================================
        # Subplot 2: Aerodynamic performance (L/D vs AoA)
        # ==================================================
        plt.subplot(1, 2, 2)

        # Recreate angle of attack vector
        alphas = np.linspace(self.lower_alpha, self.upper_alpha, self.n_alphas)

        # Calculate efficiency curve (avoid division by zero)
        valid_cd = np.where(self._current_cd_sweep > 1e-5, self._current_cd_sweep, 1e-5)
        efficiency_sweep = self._current_cl_sweep / valid_cd

        # Plot complete efficiency curve
        plt.plot(
            alphas, efficiency_sweep, color="green", linewidth=2, label="L/D Curve"
        )

        # Mark maximum efficiency point
        max_idx = np.argmax(efficiency_sweep)
        max_alpha = alphas[max_idx]
        max_eff = efficiency_sweep[max_idx]

        plt.plot(
            max_alpha,
            max_eff,
            "ro",
            markersize=8,
            label=f"Max L/D: {max_eff:.1f} @ {max_alpha:.1f}°",
        )

        # Calculate latent space distance
        latent_distance = np.linalg.norm(self._current_z)

        plt.title(
            f"Performance | L/D: {self._current_efficiency:.2f} | Latent Dist: {latent_distance:.3f}"
        )
        plt.xlabel("Angle of Attack (degrees)")
        plt.ylabel("Efficiency (L/D)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper right")

        # Lock polar axes for consistent display
        plt.xlim(self.lower_alpha, self.upper_alpha)
        plt.ylim(0, 150)

        # ==================================================
        # Update display
        # ==================================================
        plt.tight_layout()  # Prevent subplot overlap
        plt.pause(0.01)

    def close(self):
        # Disable interactive mode and close all figures to prevent memory leaks
        plt.ioff()
        plt.close("all")
