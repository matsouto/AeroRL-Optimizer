# TODO: Test Mach number and Reynolds number calculation
# TODO: Extract base class from this and other environments
# TODO: Comparar resultados da validação action range menor vs maior

import joblib
import gymnasium as gym
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import isacalc as isa
from pathlib import Path
from gymnasium import spaces
from aerosandbox import Airfoil
from scipy.interpolate import interp1d
from src.helpers import cst_to_coords

from src.airfoil import airfoil_modifications


class MultiStageOptEnv(gym.Env):
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
        self._current_stage_efficiency = 0.0
        self._analysis_confidence = 0.0
        self._current_target_cl = 0.0
        self._current_stage = -1

        # Initialize flight conditions
        self._re = 0.0
        self._mach = 0.0
        self.altitude = 0.0
        self.velocity = 0.0

        # Step counter
        self._current_step = 0

        # Action space: adjustments to the latent vector
        self.action_space = spaces.Box(
            low=-self.action_range,
            high=self.action_range,
            shape=(self.latent_dim,),
            dtype=np.float32,
        )

        # Observation space: latent vector with flight conditions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.latent_dim + 3,),
            dtype=np.float32,
        )

        # Initialize scaler
        self.scaler = joblib.load(scaler_path)

        # Initialize ONNX runtime session for the decoder
        self.session = ort.InferenceSession(decoder_path)
        self.input_name = self.session.get_inputs()[0].name

    def _get_obs(self):
        # Scale Reynolds number to prevent gradient explosion
        re_norm = self._re / 1e6

        return np.concatenate(
            [
                self._current_z,
                [re_norm],
                [self._current_target_cl],
                [float(self._current_stage)],
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {
            "cl_sweep": self._current_cl_sweep,
            "cd_sweep": self._current_cd_sweep,
            "target_cl": self._current_target_cl,
        }

    def _get_coords(self):
        # Run ONNX inference on decoder
        outputs = self.session.run(
            None, {self.input_name: self._current_z.reshape(1, -1)}
        )

        # ONNX returns normalized weight and pressure tensors
        w_norm = outputs[0]
        p_norm = outputs[1]

        # Denormalize using the scaler
        w_phys, p_phys = self.scaler.inverse_transform(w_norm, p_norm)

        # Generate CST coordinates
        x_coords, y_coords = cst_to_coords(w_phys[0], p_phys[0], n_points=100)

        coords = np.stack((x_coords, y_coords), axis=-1)

        return coords

    def _get_flight_conditions(self):
        """
        Sample flight conditions for the current mission phase.
        """
        mass = 40  # kg
        weight = mass * 9.81
        wing_area = 1.5  # m^2
        coord = 0.4  # m

        match self._current_stage:
            case 0:  # Ascent
                altitude = self.np_random.uniform(0, 1500)
                velocity = self.np_random.uniform(16.0, 25.0)
            case 1:  # High-speed dash
                altitude = self.np_random.uniform(500, 1500)
                velocity = self.np_random.uniform(25.0, 45.0)
            case 2:  # Loiter
                altitude = self.np_random.uniform(500, 1000)
                velocity = self.np_random.uniform(15.0, 25.0)
            case 3:  # Cruise/return
                altitude = self.np_random.uniform(500, 1500)
                velocity = self.np_random.uniform(20.0, 35.0)

        self.altitude = altitude
        self.velocity = velocity

        # Calculate atmospheric properties at altitude
        _, _, _, _rho, _sound_speed, _mu = isa.Atmosphere().calculate(altitude)

        _current_target_cl = weight / (0.5 * _rho * velocity**2 * wing_area)
        _re = _rho * velocity * coord / _mu
        _mach = velocity / _sound_speed

        return _re, _mach, _current_target_cl

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
                alpha=np.linspace(self.lower_alpha, self.upper_alpha, self.n_alphas),
                Re=self._re,
                mach=self._mach,
                model_size="xlarge",
            )

            self._current_cl_sweep = aero["CL"]
            self._current_cd_sweep = aero["CD"]
            self._analysis_confidence = aero["analysis_confidence"]

            _confidence_threshold = 0.30
            valid_mask = self._analysis_confidence >= _confidence_threshold

            if not np.any(valid_mask):
                reward = -1.0  # Fixed penalty for low confidence
                self._current_stage_efficiency = 0.0
            else:
                valid_cl = self._current_cl_sweep[valid_mask]
                valid_cd = self._current_cd_sweep[valid_mask]

                # Protect against division by zero with minimum CD and CL values
                safe_cd = np.maximum(valid_cd, 0.005)
                safe_cl = np.maximum(valid_cl, 0.001)

                # Calculate airfoil thickness
                max_thickness = af.max_thickness()

                # Penalty for too-thin airfoils
                thickness_penalty = 0.0
                if max_thickness < 0.12:
                    diff = 0.12 - max_thickness
                    thickness_penalty = diff * 1000.0

                # Penalty for deviating from target CL
                cl_penalty = 0.0
                _cl_error = abs(valid_cl - self._current_target_cl)
                cl_penalty = np.exp(-15.0 * (_cl_error**2))

                match self._current_stage:
                    case 0:
                        # Phase 0: Ascent
                        # Maximize lift-to-drag ratio for altitude gain
                        self._current_stage_efficiency_arr = valid_cl / safe_cd

                    case 1:
                        # Phase 1: High-speed dash
                        # Maximize penetration (minimize drag): maximize 1/CD
                        self._current_stage_efficiency_arr = 1.0 / safe_cd

                    case 2:
                        # Phase 2: Loiter
                        # Maximize endurance: CL^1.5 / CD for electric platforms
                        self._current_stage_efficiency_arr = (safe_cl**1.5) / safe_cd

                    case 3:
                        # Phase 3: Cruise return
                        # Maximize range: CL / CD for efficient flight
                        self._current_stage_efficiency_arr = valid_cl / safe_cd

                reward_arr = (
                    self._current_stage_efficiency_arr * cl_penalty - thickness_penalty
                )

                _best_idx = np.argmax(reward_arr)
                self._current_stage_efficiency = self._current_stage_efficiency_arr[
                    _best_idx
                ]
                raw_reward = float(reward_arr[_best_idx])
                # Scale and clip reward to prevent extreme values, but allow positive rewards to grow without an upper bound
                reward = np.clip(raw_reward / 100.0, -2.0, np.inf)

        except Exception as e:
            reward = -1.0
            self._current_stage_efficiency = 0.0

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
        self._current_stage = self.np_random.integers(0, 4)
        self._re, self._mach, self._current_target_cl = self._get_flight_conditions()

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

        # ==================================================
        # NOVO: Lógica de interceptação do Target CL
        # ==================================================
        alphas = np.linspace(self.lower_alpha, self.upper_alpha, self.n_alphas)

        # Encontra o índice onde o CL gerado chegou mais perto do CL alvo
        closest_idx = np.argmin(
            np.abs(self._current_cl_sweep - self._current_target_cl)
        )
        achieved_cl = self._current_cl_sweep[closest_idx]
        achieved_alpha = alphas[closest_idx]
        cl_error = abs(achieved_cl - self._current_target_cl)

        # Define a cor do marcador baseado no erro (Verde = < 0.05, Amarelo = < 0.15, Vermelho = Erro grave)
        if cl_error < 0.05:
            marker_color = "lime"
        elif cl_error < 0.15:
            marker_color = "gold"
        else:
            marker_color = "red"

        # Enable interactive mode
        plt.ion()

        # Create or reuse figure with horizontal aspect ratio
        if not plt.fignum_exists(1):
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5), num=1)
        else:
            self.fig.clf()
            self.ax1 = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)

        # ==================================================
        # Subplot 1: Airfoil geometry and Phase Info
        # ==================================================
        self.ax1.plot(x_coords, y_coords, color="#1f77b4", linewidth=2)
        self.ax1.fill(x_coords, y_coords, color="#1f77b4", alpha=0.15)

        phase_info = {
            0: {"name": "ASCENT", "objective": "Maximize L/D"},
            1: {"name": "DASH", "objective": "Minimize CD (Penetration)"},
            2: {"name": "LOITER", "objective": "Maximize CL^1.5 / CD (Endurance)"},
            3: {"name": "CRUISE", "objective": "Maximize CL / CD (Range)"},
        }

        current_phase = int(self._current_stage)
        p_name = phase_info.get(current_phase, {}).get("name", "UNKNOWN")
        p_obj = phase_info.get(current_phase, {}).get("objective", "UNKNOWN")

        latent_distance = np.linalg.norm(self._current_z)

        # Texto atualizado com o Target e o Erro
        info_text = (
            f"Phase {current_phase}: {p_name}\n"
            f"Objective: {p_obj}\n"
            f"---------------------------\n"
            f"Target CL : {self._current_target_cl:.2f}\n"
            f"Achieved  : {achieved_cl:.2f} (Error: {cl_error:.3f})\n"
            f"---------------------------\n"
            f"Thickness : {max_thickness*100:.2f}%\n"
            f"Camber    : {camber*100:.2f}%\n"
            f"LE Radius : {le_radius:.4f}\n"
            f"Latent Dist: {latent_distance:.2f}"
        )

        props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
        self.ax1.text(
            0.03,
            0.95,
            info_text,
            transform=self.ax1.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=props,
            family="monospace",
        )

        self.ax1.set_title("Real-Time Morphing Geometry", fontweight="bold")
        self.ax1.set_xlabel("Chord (X)")
        self.ax1.set_ylabel("Thickness (Y)")
        self.ax1.grid(True, linestyle="--", alpha=0.6)
        self.ax1.axis("equal")
        self.ax1.set_xlim(-0.05, 1.05)
        self.ax1.set_ylim(-0.35, 0.35)

        # ==================================================
        # Subplot 2: Aerodynamic performance (CL and CD vs AoA)
        # ==================================================
        color_cl = "tab:green"
        self.ax2.plot(
            alphas,
            self._current_cl_sweep,
            color=color_cl,
            linewidth=2.5,
            label="CL Curve",
        )

        # Linha do Target CL (Mudei para preto e tracejado para dar contraste)
        self.ax2.axhline(
            y=self._current_target_cl,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Target CL",
        )

        # O Marcador visual de Sucesso/Falha no ponto exato
        self.ax2.plot(
            achieved_alpha,
            achieved_cl,
            marker="o",
            color=marker_color,
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1.5,
            label=f"Achieved @ {achieved_alpha:.1f}°",
            zorder=5,
        )

        # Uma linha vertical fina indicando o Ângulo de Ataque necessário
        self.ax2.vlines(
            x=achieved_alpha,
            ymin=min(self._current_cl_sweep),
            ymax=achieved_cl,
            color=marker_color,
            linestyle=":",
            alpha=0.8,
        )

        self.ax2.set_xlabel("Angle of Attack (degrees)")
        self.ax2.set_ylabel("Lift Coefficient (CL)", color=color_cl, fontweight="bold")
        self.ax2.tick_params(axis="y", labelcolor=color_cl)
        self.ax2.grid(True, linestyle="--", alpha=0.6)

        ax3 = self.ax2.twinx()
        color_cd = "tab:red"
        ax3.plot(
            alphas,
            self._current_cd_sweep,
            color=color_cd,
            linewidth=2.5,
            label="CD Curve",
        )
        ax3.set_ylabel("Drag Coefficient (CD)", color=color_cd, fontweight="bold")
        ax3.tick_params(axis="y", labelcolor=color_cd)

        lines_1, labels_1 = self.ax2.get_legend_handles_labels()
        lines_2, labels_2 = ax3.get_legend_handles_labels()
        self.ax2.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper left",
            framealpha=0.95,
            ncol=1,
        )

        self.ax2.set_title("Aerodynamic Polar Curves", fontweight="bold")

        # ==================================================
        # Update display
        # ==================================================
        plt.tight_layout()
        plt.pause(0.01)

    def close(self):
        # Disable interactive mode and close all figures to prevent memory leaks
        plt.ioff()
        plt.close("all")
