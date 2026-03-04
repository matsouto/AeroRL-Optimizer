# TESTAR AGORA NO SKETCH E DEPOIS RODAR O TREINAMENTO DE VERDAEDE

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
            confidence_sweep = aero["analysis_confidence"]

            # MÁSCARA DE SEGURANÇA MÍNIMA:
            valid_mask = confidence_sweep >= 0.30

            if not np.any(valid_mask):
                # Se o perfil é tão ruim que não tem 1 ângulo confiável
                self._current_efficiency = -20.0
                reward = -20.0
            else:
                # Usa a eficiência BRUTA (sem penalidades) para os pontos válidos
                valid_cl = self._current_cl_sweep[valid_mask]
                valid_cd = self._current_cd_sweep[valid_mask]

                self._current_efficiency = float(np.max(valid_cl / valid_cd))
                reward = self._current_efficiency

        except Exception as e:
            self._current_cl_sweep = np.zeros(self.n_alphas)
            self._current_cd_sweep = np.zeros(self.n_alphas)
            self._current_efficiency = -50.0
            reward = -50.0

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

        # 2. Configurar o modo interativo do Matplotlib (para não travar o loop)
        plt.ion()

        # 3. Limpar o frame anterior para desenhar o novo
        plt.clf()

        # 4. Desenhar o aerofólio
        plt.plot(x_coords, y_coords, color="blue", linewidth=2)
        plt.fill(x_coords, y_coords, color="blue", alpha=0.15)

        # 5. Formatação do Gráfico para ficar com cara de software profissional
        # Exibe a eficiência na tela em tempo real
        plt.title(
            f"Otimização em Andamento | Eficiência (L/D) Máx: {self._current_efficiency:.2f}"
        )
        plt.xlabel("Corda (X)")
        plt.ylabel("Espessura (Y)")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Travar os eixos é CRÍTICO para a animação não ficar pulando
        plt.axis("equal")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.25, 0.25)

        # Atualiza a tela rapidamente (10 milissegundos) e devolve o controle ao script
        plt.pause(0.01)

    def close(self):
        # Desliga o modo interativo e fecha todas as janelas do matplotlib
        # Isso impede que o seu computador fique sem RAM após longos testes
        plt.ioff()
        plt.close("all")
