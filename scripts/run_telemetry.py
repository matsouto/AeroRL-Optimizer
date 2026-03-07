import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.enviroments import MultiStageOptEnv

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
import isacalc as isa

RUN_ID = "20260307-145624"

PROJECT_PATH = Path(__file__).resolve().parent.parent
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

MODEL_PATH = MODELS_PATH / RUN_ID / "best_model" / "best_model.zip"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Please check the path and try again."
    )
else:
    print(f"Loaded RL model from: {MODEL_PATH}")

# ==========================================
# CONFIGURAÇÕES DA TELEMETRIA
# ==========================================
TOTAL_STEPS = 400  # Duração total do voo simulado


def run_telemetry_dashboard():
    print("\n🛰️ Iniciando Sistema de Telemetria e Voo Contínuo...")

    env = MultiStageOptEnv(
        scaler_path=SCALER_PATH,
        decoder_path=DECODER_PATH,
        max_episode_steps=TOTAL_STEPS,  # Expandimos o limite para o voo completo
    )

    model = PPO.load(MODEL_PATH)
    obs, info = env.reset()

    # Zera a VAE para o perfil decolar num formato neutro e limpo
    env.unwrapped._current_z = np.zeros(env.unwrapped.latent_dim, dtype=np.float32)

    # ==========================================
    # DEFINIÇÃO DO PERFIL DE VOO CONTÍNUO (Timeline)
    # ==========================================
    # Passos-chave: 0 (Decolagem) -> 100 (Fim Subida) -> 200 (Fim Dash) -> 300 (Fim Loiter) -> 400 (Retorno)
    keyframes = [0, 100, 150, 200, 250, 300, 350, 400]
    velocities = [16.0, 25.0, 45.0, 40.0, 15.0, 15.0, 25.0, 25.0]  # m/s
    altitudes = [0.0, 1000.0, 1500.0, 1500.0, 800.0, 800.0, 500.0, 500.0]  # metros

    # Variáveis de Histórico para os gráficos da direita
    hist_steps = []
    hist_eff = []
    hist_alt = []
    hist_vel = []

    # ==========================================
    # SETUP DO DASHBOARD MATPLOTLIB (Alta Performance)
    # ==========================================
    plt.ion()
    fig = plt.figure(figsize=(15, 7))
    fig.canvas.manager.set_window_title("UAV Morphing Wing - Telemetria ao Vivo")
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1])

    # Painel Esquerdo: Asa
    ax_wing = fig.add_subplot(gs[:, 0])
    ax_wing.set_title(
        "Geometria Dinâmica (Morphing em Tempo Real)", fontsize=14, fontweight="bold"
    )
    ax_wing.set_xlabel("Corda (X)")
    ax_wing.set_ylabel("Espessura (Y)")
    ax_wing.grid(True, linestyle="--", alpha=0.6)
    ax_wing.axis("equal")
    ax_wing.set_xlim(-0.2, 1.2)
    ax_wing.set_ylim(-0.35, 0.35)

    (line_upper,) = ax_wing.plot([], [], color="#1f77b4", linewidth=3)
    (line_lower,) = ax_wing.plot([], [], color="#1f77b4", linewidth=3)
    text_hud = ax_wing.text(
        0.03,
        0.95,
        "",
        transform=ax_wing.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.8, edgecolor="none"),
        color="lime",
    )

    # Painel Direito Superior: Perfil da Missão
    ax_mission = fig.add_subplot(gs[0, 1])
    ax_mission.set_title("Perfil de Voo da Missão", fontsize=12, fontweight="bold")
    ax_mission.set_xlim(0, TOTAL_STEPS)
    ax_mission.set_ylim(0, 1600)
    ax_mission.set_ylabel("Altitude (m)", color="tab:blue", fontweight="bold")
    (line_alt,) = ax_mission.plot([], [], color="tab:blue", linewidth=2)
    ax_mission.grid(True, linestyle=":", alpha=0.6)

    ax_vel = ax_mission.twinx()
    ax_vel.set_ylim(10, 50)
    ax_vel.set_ylabel("Velocidade (m/s)", color="tab:red", fontweight="bold")
    (line_vel,) = ax_vel.plot([], [], color="tab:red", linestyle="--", linewidth=2)

    # Painel Direito Inferior: Eficiência Aerodinâmica
    ax_eff = fig.add_subplot(gs[1, 1])
    ax_eff.set_title("Eficiência Otimizada pela IA", fontsize=12, fontweight="bold")
    ax_eff.set_xlim(0, TOTAL_STEPS)
    ax_eff.set_ylim(0, 300)  # Ajuste este teto se os números subirem muito
    ax_eff.set_xlabel("Tempo de Voo (Passos)")
    ax_eff.set_ylabel("Recompensa Bruta", color="tab:green", fontweight="bold")
    (line_eff,) = ax_eff.plot([], [], color="tab:green", linewidth=2.5)
    ax_eff.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    # ==========================================
    # LOOP DE VOO CONTÍNUO
    # ==========================================
    print("Voo Iniciado! Assista à janela gráfica...")

    for step in range(TOTAL_STEPS):
        # 1. Determina a Fase da Missão dinamicamente
        if step < 100:
            current_stage = 0
            phase_name = "ASCENT (Subida)"
        elif step < 200:
            current_stage = 1
            phase_name = "DASH (Penetração Rápida)"
        elif step < 300:
            current_stage = 2
            phase_name = "LOITER (Busca Lenta)"
        else:
            current_stage = 3
            phase_name = "CRUISE (Retorno)"

        # 2. Interpola Suavemente as Condições Físicas
        current_v = np.interp(step, keyframes, velocities)
        current_h = np.interp(step, keyframes, altitudes)

        # Calcula a Atmosfera ISA na altitude exata
        _, _, _, rho, sound_speed, mu = isa.Atmosphere().calculate(current_h)

        mass = 40.0  # kg
        weight = mass * 9.81
        wing_area = 1.5
        coord = 0.4

        # Equações Aerodinâmicas
        dyn_press = 0.5 * rho * current_v**2
        target_cl = weight / (dyn_press * wing_area)
        reynolds = (rho * current_v * coord) / mu
        mach = current_v / sound_speed

        # 3. "INJEÇÃO" DE FÍSICA NO AMBIENTE
        # Atualizamos os valores internos do ambiente antes de pedir uma ação para a IA
        env.unwrapped._current_stage = current_stage
        env.unwrapped.altitude = current_h
        env.unwrapped.velocity = current_v
        env.unwrapped._re = reynolds
        env.unwrapped._mach = mach
        env.unwrapped._current_target_cl = target_cl

        # Atualizamos o vetor de observação para o agente enxergar a nova realidade
        obs = env.unwrapped._get_obs()

        # 4. A IA pilota e altera a asa (usando action_range real de forma fluida)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # O Piloto Automático trava a ação se a análise do NeuralFoil perder a confiança
        if env.unwrapped._current_stage_efficiency < 0:
            env.unwrapped._current_stage_efficiency = hist_eff[-1] if hist_eff else 0

        # 5. ATUALIZAÇÃO DO DASHBOARD (Gravação em Tempo Real)
        hist_steps.append(step)
        hist_alt.append(current_h)
        hist_vel.append(current_v)
        hist_eff.append(env.unwrapped._current_stage_efficiency)

        # Pega a nova geometria gerada pelo ONNX
        coords = env.unwrapped._get_coords()
        x, y = coords[:, 0], coords[:, 1]

        # Atualiza a Asa (Para não piscar a tela, alteramos apenas os dados)
        half = len(x) // 2
        line_upper.set_data(x[:half], y[:half])
        line_lower.set_data(x[half:], y[half:])

        # Atualiza as Linhas
        line_alt.set_data(hist_steps, hist_alt)
        line_vel.set_data(hist_steps, hist_vel)
        line_eff.set_data(hist_steps, hist_eff)

        # Atualiza o HUD (Heads-Up Display)
        hud_text = (
            f"FASE ATUAL : {phase_name}\n"
            f"---------------------------------\n"
            f"Altitude   : {current_h:4.0f} m\n"
            f"Velocidade : {current_v:4.1f} m/s\n"
            f"Densidade  : {rho:.3f} kg/m³\n"
            f"---------------------------------\n"
            f"Target CL  : {target_cl:.3f}\n"
            f"Eficiência : {hist_eff[-1]:.1f}\n"
        )
        text_hud.set_text(hud_text)

        # Renderiza o quadro (Frame) e avança
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)  # Atraso de 10ms para os olhos humanos conseguirem acompanhar

    print("\n✅ Missão Simulada com Sucesso!")
    plt.ioff()
    plt.show()  # Mantém o gráfico final aberto


if __name__ == "__main__":
    run_telemetry_dashboard()
