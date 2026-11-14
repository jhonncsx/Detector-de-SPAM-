"""
Calculadora de Engajamento de TikToker - Streamlit + TensorFlow (arquivo único)

Como rodar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Como funciona (resumo):
    1. O app gera uma base SIMULADA de vídeos (categoria, hashtags, alcance).
    2. Treina uma rede neural de regressão (TensorFlow/Keras) para prever
       o alcance a partir da categoria e do número de hashtags.
    3. O usuário escolhe categoria e hashtags em caixas de seleção e o app
       mostra o alcance previsto com um alerta: Flopado, Regular ou Viral.
"""

import os

# Reduz mensagens de log do TensorFlow (precisa vir ANTES do import do tensorflow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# 1) CONFIGURAÇÕES GERAIS
# ---------------------------------------------------------------------------
SEED = 42                # Semente fixa: resultados reprodutíveis
N_VIDEOS = 2000          # Tamanho da base simulada
EPOCAS = 60              # Quantas vezes o modelo "revê" os dados no treino
MAX_HASHTAGS = 15        # Máximo de hashtags oferecido na tela

LIMITE_FLOP = 10_000     # Abaixo disso: Flopado
LIMITE_VIRAL = 100_000   # Acima disso: Viral

# Alcance "típico" de cada categoria (valores fictícios, só para a simulação)
ALCANCE_BASE = {
    "Humor": 60_000,
    "Dança": 90_000,
    "Beleza": 30_000,
    "Games": 20_000,
    "Educação": 8_000,
    "Culinária": 40_000,
    "Fitness": 25_000,
}
CATEGORIAS = list(ALCANCE_BASE.keys())


# ---------------------------------------------------------------------------
# 2) BASE SIMULADA
# ---------------------------------------------------------------------------
def gerar_base_simulada() -> pd.DataFrame:
    """
    Cria vídeos fictícios seguindo regras simples:
      - cada categoria tem um alcance base diferente;
      - o número ideal de hashtags fica entre 3 e 6 (pico em ~4,5);
      - acima de 10 hashtags o alcance cai (parece "spam" para o público);
      - um ruído aleatório deixa os dados mais realistas.
    """
    rng = np.random.default_rng(SEED)
    categorias = rng.choice(CATEGORIAS, size=N_VIDEOS)
    hashtags = rng.integers(0, MAX_HASHTAGS + 1, size=N_VIDEOS)

    base = np.array([ALCANCE_BASE[c] for c in categorias], dtype="float64")
    bonus_ideal = 0.7 * np.exp(-((hashtags - 4.5) ** 2) / (2 * 2.0**2))
    penalidade = 0.15 * np.maximum(0, hashtags - 10)
    ruido = rng.normal(0, 0.35, size=N_VIDEOS)

    # Trabalhamos em escala logarítmica: efeitos viram somas simples
    log_alcance = np.log(base) + bonus_ideal - penalidade + ruido

    return pd.DataFrame(
        {
            "categoria": categorias,
            "hashtags": hashtags,
            "alcance": np.exp(log_alcance).round().astype(int),
        }
    )


# ---------------------------------------------------------------------------
# 3) TREINO DO MODELO
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparando o modelo, aguarde alguns segundos...")
def treinar_modelo():
    """
    Treina a rede e devolve (modelo, média_log, desvio_log, erro_típico_%).

    @st.cache_resource faz o Streamlit treinar uma única vez e reaproveitar
    o modelo nas próximas interações.
    """
    keras.utils.set_random_seed(SEED)
    df = gerar_base_simulada()

    # --- Alvo em escala log e padronizado ---------------------------------
    # Alcances variam de milhares a centenas de milhares. Com o logaritmo e a
    # padronização (média 0, desvio 1) o modelo aprende de forma bem mais estável.
    y_log = np.log(df["alcance"].to_numpy(dtype="float64"))
    media_log, desvio_log = float(y_log.mean()), float(y_log.std())
    y = ((y_log - media_log) / desvio_log).astype("float32")

    # --- Divisão 80% treino / 20% teste -----------------------------------
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(df))
    corte = int(len(df) * 0.8)
    treino, teste = indices[:corte], indices[corte:]

    # O Keras 3 não aceita arrays de texto do NumPy: convertemos a categoria
    # para tensor tf.string, no formato (n, 1). Hashtags viram float32 (n, 1).
    def para_tensores(idx):
        cats = tf.constant([str(c) for c in df["categoria"].to_numpy()[idx]])
        tags = df["hashtags"].to_numpy(dtype="float32")[idx]
        return {
            "categoria": tf.reshape(cats, (-1, 1)),
            "hashtags": tf.constant(tags.reshape(-1, 1)),
        }

    x_treino, x_teste = para_tensores(treino), para_tensores(teste)
    ds_treino = (
        tf.data.Dataset.from_tensor_slices((x_treino, y[treino]))
        .shuffle(len(treino), seed=SEED)
        .batch(32)
    )

    # --- Pré-processamento dentro do modelo -------------------------------
    # Categoria (texto) -> vetor one-hot. StringLookup faz o papel de
    # "StringLookup + CategoryEncoding" em uma única camada.
    entrada_cat = keras.Input(shape=(1,), dtype=tf.string, name="categoria")
    cat_codificada = keras.layers.StringLookup(
        vocabulary=CATEGORIAS, output_mode="one_hot"
    )(entrada_cat)

    # Hashtags -> normalizadas (média 0, desvio 1) com a camada Normalization.
    # O adapt() aprende média e desvio usando SÓ os dados de treino.
    entrada_tags = keras.Input(shape=(1,), dtype=tf.float32, name="hashtags")
    normalizador = keras.layers.Normalization(axis=-1)
    normalizador.adapt(df["hashtags"].to_numpy(dtype="float32")[treino].reshape(-1, 1))
    tags_normalizadas = normalizador(entrada_tags)

    # --- Rede neural de regressão -----------------------------------------
    # As camadas ocultas permitem aprender que existe um número "ideal" de
    # hashtags (e não apenas "mais é melhor" ou "menos é melhor").
    x = keras.layers.Concatenate()([cat_codificada, tags_normalizadas])
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    saida = keras.layers.Dense(1)(x)  # sem ativação: prevê um número livre
    modelo = keras.Model([entrada_cat, entrada_tags], saida)

    modelo.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    modelo.fit(ds_treino, epochs=EPOCAS, verbose=0)

    # --- Erro no conjunto de teste ----------------------------------------
    # O erro (MAE) está em escala log padronizada; convertemos para "% típico".
    previsto = modelo(x_teste, training=False).numpy().flatten()
    erro_log = float(np.mean(np.abs(previsto - y[teste])) * desvio_log)
    erro_percentual = float(np.exp(erro_log) - 1)

    return modelo, media_log, desvio_log, erro_percentual


def prever_alcance(modelo, media_log, desvio_log, categorias, hashtags) -> np.ndarray:
    """Recebe listas de categorias/hashtags e devolve os alcances previstos."""
    entrada = {
        "categoria": tf.constant([[c] for c in categorias]),
        "hashtags": tf.constant([[float(h)] for h in hashtags]),
    }
    saida = modelo(entrada, training=False).numpy().flatten()
    return np.exp(saida * desvio_log + media_log)  # volta da escala log


# ---------------------------------------------------------------------------
# 4) FUNÇÕES AUXILIARES DA INTERFACE
# ---------------------------------------------------------------------------
def formatar_numero(valor: float) -> str:
    """Formata no padrão brasileiro: 12345 -> '12.345'."""
    return f"{valor:,.0f}".replace(",", ".")


def classificar(alcance: float) -> str:
    """Transforma o número previsto em uma decisão: Flopado, Regular ou Viral."""
    if alcance < LIMITE_FLOP:
        return "Flopado"
    if alcance > LIMITE_VIRAL:
        return "Viral"
    return "Regular"


# ---------------------------------------------------------------------------
# 5) INTERFACE (STREAMLIT)
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Calculadora de Engajamento",
        page_icon="📈",
        layout="centered",
    )

    modelo, media_log, desvio_log, erro_pct = treinar_modelo()

    # ----- Barra lateral: informações do modelo -----
    with st.sidebar:
        st.header("Sobre o modelo")
        st.metric("Erro típico no teste", f"± {erro_pct:.0%}")
        st.caption(
            f"Treinado com {N_VIDEOS} vídeos simulados (80% treino, 20% teste). "
            "O erro indica a diferença típica entre o alcance previsto e o "
            "alcance real da base de teste."
        )
        st.divider()
        st.caption("Critérios do alerta")
        st.caption(
            f"🔴 Flopado: abaixo de {formatar_numero(LIMITE_FLOP)}  \n"
            f"🟡 Regular: de {formatar_numero(LIMITE_FLOP)} a "
            f"{formatar_numero(LIMITE_VIRAL)}  \n"
            f"🟢 Viral: acima de {formatar_numero(LIMITE_VIRAL)}"
        )

    # ----- Área principal -----
    st.title("📈 Calculadora de Engajamento de TikToker")
    st.write(
        "Selecione a categoria do vídeo e a quantidade de hashtags para "
        "estimar o alcance antes de publicar."
    )
    st.info(
        "Atenção: esta previsão utiliza dados simulados e serve como "
        "demonstração. Não representa o desempenho real na plataforma."
    )

    col1, col2 = st.columns(2)
    categoria = col1.selectbox("Categoria do vídeo", CATEGORIAS)
    hashtags = col2.selectbox(
        "Quantidade de hashtags", list(range(MAX_HASHTAGS + 1)), index=4
    )

    if st.button("Calcular alcance", type="primary"):
        alcance = float(
            prever_alcance(modelo, media_log, desvio_log, [categoria], [hashtags])[0]
        )
        status = classificar(alcance)

        st.metric("Alcance previsto (visualizações)", formatar_numero(alcance))

        if status == "Viral":
            st.success("🚀 **VIRAL**: grande potencial de alcance para este vídeo.")
        elif status == "Regular":
            st.warning("📊 **REGULAR**: desempenho dentro do esperado.")
        else:
            st.error("📉 **FLOPADO**: baixo potencial de alcance neste formato.")

        # Gráfico: como o alcance muda conforme o número de hashtags
        todas = list(range(MAX_HASHTAGS + 1))
        curva = prever_alcance(
            modelo, media_log, desvio_log, [categoria] * len(todas), todas
        )
        melhor = todas[int(np.argmax(curva))]
        st.subheader(f"Alcance previsto por nº de hashtags ({categoria})")
        st.line_chart(
            pd.DataFrame({"Alcance previsto": curva}, index=pd.Index(todas, name="Hashtags"))
        )
        st.caption(f"Para esta categoria, o modelo indica melhor resultado com cerca de {melhor} hashtags.")

    st.divider()
    st.caption(
        "A previsão é uma estimativa de um modelo de IA e deve apoiar, "
        "não substituir, a análise da equipe de marketing."
    )


if __name__ == "__main__":
    main()