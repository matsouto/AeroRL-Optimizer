import joblib
import gymnasium as gym
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path
from gymnasium import spaces
from aerosandbox import Airfoil
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

        # Initialize the current latent vector and coefficients to default values
        self._current_z = np.zeros(self.latent_dim, dtype=np.float32)
        self._current_cl_sweep = np.zeros(self.n_alphas, dtype=np.float32)
        self._current_cd_sweep = np.zeros(self.n_alphas, dtype=np.float32)
        self._current_efficiency = 0.0
        self._analysis_confidence = 0.0

        # Initialize step counter
        self._current_step = 0

        # The action consists of adjustments to the latent vector
        self.action_space = spaces.Box(
            low=-self.action_range,
            high=self.action_range,
            shape=(self.latent_dim,),
            dtype=np.float32,
        )

        # The observation consists of the current latent vector
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
        # 4. Rodar Inferência
        outputs = self.session.run(
            None, {self.input_name: self._current_z.reshape(1, -1)}
        )

        # O ONNX retorna os tensores crus (w_norm e p_norm)
        w_norm = outputs[0]
        p_norm = outputs[1]

        # 5. Desnormalizar os dados
        w_phys, p_phys = self.scaler.inverse_transform(w_norm, p_norm)

        # 6. Gerar Coordenadas CST
        x_coords, y_coords = cst_to_coords(w_phys[0], p_phys[0], n_points=100)

        coords = np.stack((x_coords, y_coords), axis=-1)

        return coords

    def _get_airfoil_characteristics(self, x_coords, y_coords):
        # 1. Normalize by Chord (Assumes LE is at 0 and TE is at max X)
        chord = np.max(x_coords) - np.min(x_coords)
        xn = x_coords / chord
        yn = y_coords / chord

        # 2. Split into Upper and Lower (Requires points to be ordered LE -> TE)
        # This is a simplified split; professional tools use interp1d to align X-stations
        half = len(xn) // 2
        y_upper = yn[:half]
        y_lower = yn[half:]

        # 3. Correct Thickness & Camber Logic
        # We use the absolute difference between surfaces at the same X-stations
        thickness_dist = np.abs(y_upper - y_lower)
        max_thickness = np.max(thickness_dist) * 100  # % of chord

        # Camber line is the average of upper and lower surfaces
        camber_line = (y_upper + y_lower) / 2
        max_camber = np.max(np.abs(camber_line)) * 100  # % of chord

        # 4. LE Radius (Parabolic fit approximation)
        # A better approx uses the second derivative at the nose
        # For now, let's just fix your distance logic to be more descriptive
        le_step_size = np.sqrt((xn[1] - xn[0]) ** 2 + (yn[1] - yn[0]) ** 2)

        return max_thickness, max_camber, le_step_size

    def step(self, action):
        self._current_step += 1

        # Update the current latent vector based on the action
        self._current_z += action

        # Clip the latent vector to stay within bounds
        self._current_z = np.clip(
            self._current_z, -self.latent_range, self.latent_range
        )

        # Obter as coordenadas do perfil a partir do decoder
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
                reward = -1.0  # Penalidade fixa e pequena
                self._current_efficiency = 0.0
            else:
                valid_cl = self._current_cl_sweep[valid_mask]
                valid_cd = self._current_cd_sweep[valid_mask]

                # 1. Proteção contra divisão por zero/valores ínfimos
                # Um CD menor que 0.001 é raríssimo em condições reais
                safe_cd = np.maximum(valid_cd, 0.005)

                # 2. Cálculo da eficiência com Teto (Clipping)
                # Mesmo o melhor aerofólio do mundo dificilmente passa de 200
                eff_sweep = valid_cl / safe_cd
                raw_eff = float(np.max(eff_sweep))
                self._current_efficiency = np.clip(raw_eff, 0, 250)

                # Calcula a espessura máxima aproximada (diferença entre Y máximo e mínimo)
                coords = self._get_coords()
                x_coords, y_coords = coords[:, 0], coords[:, 1]
                max_thickness, _, _ = self._get_airfoil_characteristics(
                    x_coords, y_coords
                )
                max_thickness_float = max_thickness / 100.0

                thickness_penalty = 0.0

                # Ensure max_thickness is a float 0.12 (for 12%) or 0.08 (for 8%)
                if max_thickness_float < 0.10:
                    diff = 0.10 - max_thickness_float
                    # Use a squared penalty for a smoother 'soft-wall'
                    thickness_penalty = (diff**2) * 1000.0

                self.reward = (self._current_efficiency - thickness_penalty) / 100.0

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

        # Get airfoil characteristics
        max_thickness, camber, le_radius = self._get_airfoil_characteristics(
            x_coords, y_coords
        )

        # 1. Configurar o modo interativo
        plt.ion()

        # Se a figura não existir, cria uma com proporção retangular (mais larga)
        if not plt.fignum_exists(1):
            plt.figure(1, figsize=(12, 5))

        # 2. Limpar o frame anterior
        plt.clf()

        # ==========================================
        # SUBPLOT 1: A GEOMETRIA DO AEROFÓLIO
        # ==========================================
        plt.subplot(1, 2, 1)
        plt.plot(x_coords, y_coords, color="blue", linewidth=2)
        plt.fill(x_coords, y_coords, color="blue", alpha=0.15)

        plt.title(
            f"Airfoil | Thickness: {max_thickness:.2f}% | Camber: {camber:.3f}% | LE Radius: {le_radius:.4f}"
        )
        plt.xlabel("Corda (X)")
        plt.ylabel("Espessura (Y)")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Travar os eixos para não pular
        plt.axis("equal")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.25, 0.25)

        # ==========================================
        # SUBPLOT 2: A POLAR AERODINÂMICA (Eficiência vs Alpha)
        # ==========================================
        plt.subplot(1, 2, 2)

        # Recria o vetor de alphas usado no step()
        # Ajuste self.lower_alpha, self.upper_alpha e self.n_alphas se os nomes forem diferentes no seu __init__
        alphas = np.linspace(self.lower_alpha, self.upper_alpha, self.n_alphas)

        # Calcula a eficiência para toda a curva (evitando divisão por zero)
        valid_cd = np.where(self._current_cd_sweep > 1e-5, self._current_cd_sweep, 1e-5)
        efficiency_sweep = self._current_cl_sweep / valid_cd

        # Plota a curva completa
        plt.plot(
            alphas, efficiency_sweep, color="green", linewidth=2, label="L/D Curve"
        )

        # Encontra e marca o ponto de eficiência máxima
        max_idx = np.argmax(efficiency_sweep)
        max_alpha = alphas[max_idx]
        max_eff = efficiency_sweep[max_idx]

        plt.plot(
            max_alpha,
            max_eff,
            "ro",
            markersize=8,
            label=f"Máx L/D: {max_eff:.1f} @ {max_alpha:.1f}°",
        )

        # Calculate latent distance for display
        latent_distance = np.linalg.norm(self._current_z)

        plt.title(
            f"Performance | L/D: {self._current_efficiency:.2f} | Latent Dist: {latent_distance:.3f}"
        )
        plt.xlabel("Ângulo de Ataque (Graus)")
        plt.ylabel("Eficiência (L/D)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper right")

        # Travar os eixos da polar é CRÍTICO.
        # Ajuste o ylim superior (ex: 150) dependendo do máximo que seus perfis costumam atingir
        plt.xlim(self.lower_alpha, self.upper_alpha)
        plt.ylim(0, 150)

        # ==========================================
        # ATUALIZAÇÃO DA TELA
        # ==========================================
        plt.tight_layout()  # Evita que os gráficos fiquem sobrepostos
        plt.pause(0.01)

    def close(self):
        # Desliga o modo interativo e fecha todas as janelas do matplotlib
        # Isso impede que o seu computador fique sem RAM após longos testes
        plt.ioff()
        plt.close("all")
