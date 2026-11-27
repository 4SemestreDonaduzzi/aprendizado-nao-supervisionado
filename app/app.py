"""Demonstração interativa do detector de anomalias (issue #10).

Executar a partir da raiz do projeto, depois de treinar os dois cenários:

    streamlit run app/app.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # caminhos da config e dos checkpoints são relativos à raiz

from autoencoder.data import CLASS_NAMES  # noqa: E402
from autoencoder.demo import (  # noqa: E402
    OUTCOME_LABELS,
    choose_threshold,
    classify,
    load_scenario,
    outcome,
    reconstruct_image,
    test_outcomes,
)

CONFIG = "configs/default.yaml"
SCENARIO_NAMES = {"bag": "Bag (bolsa)", "sandal": "Sandal (sandália)"}

st.set_page_config(page_title="Autoencoder: detecção de anomalias", layout="wide")


@st.cache_resource(show_spinner="Carregando o modelo e calculando os erros...")
def get_scenario(name: str):
    return load_scenario(CONFIG, name)


st.title("Detecção de anomalias com Autoencoder")
st.caption(
    "O modelo foi treinado só com as classes normais do Fashion-MNIST. Ele comprime "
    "cada imagem de 784 para 16 números e tenta reconstruí-la. Quando o erro de "
    "reconstrução passa do limiar, a imagem é marcada como anomalia."
)

# ---------- Barra lateral: escolhas do usuário ----------
with st.sidebar:
    st.header("Configuração")
    scenario = st.radio(
        "Classe tratada como anomalia",
        list(SCENARIO_NAMES),
        format_func=SCENARIO_NAMES.get,
    )
    percentile = st.radio(
        "Limiar (percentil do erro na validação)",
        [95, 99],
        format_func=lambda p: f"P{p}" + (" — mais sensível" if p == 95 else " — mais conservador"),
    )

try:
    data = get_scenario(scenario)
except FileNotFoundError as error:
    st.error(
        "Modelo treinado não encontrado. Na raiz do projeto, rode:\n\n"
        f"`python scripts/train_cenarios.py --scenario {scenario}`\n\n"
        f"Detalhe: {error}"
    )
    st.stop()

threshold = choose_threshold(data.val_mse, percentile)
outcomes = test_outcomes(data, threshold)

with st.sidebar:
    st.header("Escolher imagem do teste")
    class_options = ["Todas"] + [
        f"{CLASS_NAMES[c]}" + (" (anomalia)" if c == data.anomaly_class else "")
        for c in range(len(CLASS_NAMES))
    ]
    class_choice = st.selectbox("Classe", class_options)
    outcome_choice = st.selectbox(
        "Resultado do detector",
        ["Todos"] + list(OUTCOME_LABELS),
        format_func=lambda o: o if o == "Todos" else f"{o}: {OUTCOME_LABELS[o]}",
    )

    mask = np.ones(len(data.test_labels), dtype=bool)
    if class_choice != "Todas":
        mask &= data.test_labels == class_options.index(class_choice) - 1
    if outcome_choice != "Todos":
        mask &= outcomes == outcome_choice
    candidates = np.flatnonzero(mask)
    st.write(f"{len(candidates)} imagens com esse filtro")

if len(candidates) == 0:
    st.warning("Nenhuma imagem do teste combina com esse filtro. Mude a classe ou o resultado.")
    st.stop()

key = f"pick-{scenario}-{percentile}-{class_choice}-{outcome_choice}"
draw_again = st.sidebar.button("🎲 Sortear outra imagem")  # sempre desenhado
if key not in st.session_state or draw_again:
    st.session_state[key] = int(np.random.default_rng().choice(candidates))
index = st.session_state[key]

# ---------- Análise da imagem escolhida ----------
image = data.test_images[index]
label = int(data.test_labels[index])
result = reconstruct_image(data.model, image)
is_anomaly = label == data.anomaly_class
predicted = classify(result["mse"], threshold)
case = outcome(predicted, is_anomaly)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Classe real", CLASS_NAMES[label] + (" (anomalia)" if is_anomaly else ""))
col2.metric("Erro de reconstrução (MSE)", f"{result['mse']:.4f}")
col3.metric(f"Limiar P{percentile}", f"{threshold:.4f}")
col4.metric("Decisão do detector", "ANOMALIA" if predicted else "normal")

message = f"**{case} — {OUTCOME_LABELS[case]}.** Imagem nº {index} do conjunto de teste."
(st.success if case in ("TP", "TN") else st.error)(message)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
panels = [
    (image.reshape(28, 28), "Original", "gray", (0, 1)),
    (result["reconstruction"], "Reconstrução", "gray", (0, 1)),
    (result["error_map"], "Mapa de erro |original − reconstrução|", "inferno", (0, 1)),
]
for ax, (picture, title, cmap, (vmin, vmax)) in zip(axes, panels):
    shown = ax.imshow(picture, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.colorbar(shown, ax=axes[2], fraction=0.046, pad=0.04)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------- Onde essa imagem cai na distribuição dos erros ----------
st.subheader("Onde essa imagem fica entre todas as do teste")
fig, ax = plt.subplots(figsize=(10, 3))
bins = np.linspace(0, np.percentile(data.test_mse, 99.5), 80)
normal = data.test_labels != data.anomaly_class
ax.hist(data.test_mse[normal], bins=bins, alpha=0.6, label="Normais")
ax.hist(data.test_mse[~normal], bins=bins, alpha=0.6, label=f"Anomalias ({CLASS_NAMES[data.anomaly_class]})")
ax.axvline(threshold, color="black", linestyle="--", label=f"Limiar P{percentile}")
ax.axvline(min(result["mse"], bins[-1]), color="red", linewidth=2, label="Esta imagem")
ax.set_xlabel("Erro de reconstrução (MSE)")
ax.set_ylabel("Imagens")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

counts = {o: int((outcomes == o).sum()) for o in OUTCOME_LABELS}
st.caption(
    f"No teste inteiro, com o limiar P{percentile}: {counts['TP']} anomalias detectadas, "
    f"{counts['FN']} não detectadas, {counts['FP']} alarmes falsos e {counts['TN']} normais "
    "reconhecidas. O limiar é calculado só com a validação, que não contém anomalias."
)
