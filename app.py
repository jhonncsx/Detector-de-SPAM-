"""
Detector de Spam em E-mails - Streamlit + TensorFlow (arquivo único)

Como rodar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Como funciona (resumo):
    1. Um pequeno conjunto de mensagens de exemplo (spam / não spam) está
       embutido neste arquivo.
    2. Na primeira execução, o app treina uma rede neural simples (TensorFlow)
       e guarda o modelo em cache.
    3. O funcionário cola o texto de um e-mail e o app devolve a
       probabilidade de ser spam.
    4. Pela barra lateral é possível enviar um CSV da própria empresa
       (colunas: texto, rotulo) para retreinar o modelo com dados reais.
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
SEED = 42                # Semente: deixa o treino reprodutível
MAX_PALAVRAS = 3000      # Tamanho máximo do vocabulário do modelo
EPOCAS = 40              # Quantas vezes o modelo "revê" os dados no treino
LIMIAR_PADRAO = 0.50     # Acima disso a mensagem é classificada como spam

# ---------------------------------------------------------------------------
# 2) DADOS DE EXEMPLO (1 = spam, 0 = não spam)
#    São poucos exemplos, só para a aplicação funcionar de imediato.
#    Para uso real, envie um CSV maior pela barra lateral.
# ---------------------------------------------------------------------------
SPAM = [
    "PARABÉNS! Você ganhou um prêmio de R$ 10.000. Clique aqui agora e resgate",
    "Oferta imperdível!!! Compre 1 leve 10 só hoje, clique no link e garanta",
    "Seu empréstimo foi pré-aprovado sem consulta ao SPC. Responda já",
    "Ganhe dinheiro rápido trabalhando em casa, renda extra de R$ 5.000 por semana",
    "URGENTE: sua conta bancária foi bloqueada. Confirme seus dados neste link",
    "Emagreça 10 kg em 7 dias com este método secreto, sem dieta e sem exercício",
    "Você foi selecionado para receber um iPhone grátis. Cadastre-se agora",
    "Último aviso: seu CPF está irregular. Regularize clicando aqui",
    "Remédio barato sem receita, entrega rápida e discreta. Compre já",
    "Ganhe seguidores e curtidas de graça, resultado garantido em 24 horas",
    "Atenção, você tem um depósito pendente de R$ 3.200. Acesse e receba hoje",
    "Promoção relâmpago 90% de desconto em relógios de luxo, estoque acabando",
    "Clique aqui para cancelar sua cobrança indevida e evitar bloqueio imediato",
    "Cassino online: bônus de R$ 500 no primeiro depósito, jogue agora",
    "Investimento com lucro garantido de 300% ao mês, vagas limitadas",
    "Ficou sabendo? Você ganhou um cartão de crédito sem anuidade, aprovação imediata",
    "Seu pacote está retido na alfândega, pague a taxa neste link para liberar",
    "Mensagem exclusiva: aumente seu desempenho com este suplemento milagroso",
    "Receba agora R$ 1.500 de auxílio liberado, informe seus dados bancários",
    "Você é o visitante número 1.000.000! Resgate seu brinde antes que expire",
    "Dívidas? Limpe seu nome em 24 horas, ligue já e ganhe desconto de 90%",
    "FREE money!!! Click here to claim your prize now, limited time offer",
    "Winner! You have been selected for a free gift card, act now",
    "Cheap meds online, no prescription needed, buy now and save big",
    "Congratulations, you won the lottery. Send your bank details to receive it",
    "Confirme sua senha imediatamente ou sua conta será encerrada hoje",
    "Hot singles na sua região querem conhecer você, clique e veja as fotos",
    "Aproveite: ganhe R$ 200 de bônus ao se cadastrar, sem sair de casa",
    "Última chance de resgatar seu cupom de R$ 1.000, expira em 1 hora",
    "Método infalível para ficar rico na internet, descubra o segredo agora",
]

NAO_SPAM = [
    "Olá equipe, segue a pauta da reunião de planejamento de amanhã às 10h",
    "Bom dia! Envio em anexo o relatório de desempenho da campanha de março",
    "Podemos reagendar nossa call de alinhamento para quinta-feira à tarde?",
    "Segue a proposta comercial conforme conversamos, fico à disposição para dúvidas",
    "Lembrete: o prazo para entrega dos criativos da campanha é sexta-feira",
    "Obrigado pelo retorno. Vou revisar o contrato e respondo até o fim do dia",
    "Oi Maria, você consegue me enviar a planilha de métricas do último trimestre?",
    "Confirmo minha presença no workshop de segunda-feira na sala de reuniões",
    "Segue a nota fiscal referente ao serviço prestado em outubro",
    "Precisamos ajustar o briefing do cliente antes de apresentar a estratégia",
    "Parabéns pelo excelente resultado da equipe no último mês, bom trabalho",
    "Atualização do projeto: finalizamos a etapa de design e iniciamos o desenvolvimento",
    "Pessoal, o almoço de confraternização será na próxima quarta-feira ao meio-dia",
    "Anexei a ata da reunião de hoje, por favor confirmem se está tudo correto",
    "Tudo bem? Gostaria de marcar uma conversa para falar sobre o orçamento do ano",
    "A newsletter deste mês traz as novidades do setor e os próximos eventos",
    "Obrigado por participar do nosso webinar, segue a gravação e os slides",
    "Bom dia, o servidor ficará em manutenção no sábado das 22h às 23h",
    "Pode revisar o texto do post antes de publicarmos nas redes sociais?",
    "Segue o cronograma atualizado com as entregas previstas para o próximo mês",
    "Seu pedido foi enviado e chegará em 3 dias úteis, código de rastreio em anexo",
    "Agradeço a indicação, vou entrar em contato com o candidato ainda esta semana",
    "Hi team, please find attached the agenda for tomorrow's planning meeting",
    "Thanks for your feedback, I will update the report and send it back today",
    "Could we schedule a call next week to review the campaign results?",
    "Informamos que o recesso de fim de ano começa dia 23, com retorno em 2 de janeiro",
    "Poderia me passar o contato do responsável pelo departamento financeiro?",
    "Segue o link da videochamada de hoje às 15h, conforme combinado",
    "Os resultados da pesquisa de satisfação estão prontos, vamos analisar juntos?",
    "Olá, recebi seu currículo e gostaria de agendar uma entrevista na próxima semana",
]


def montar_dataframe_padrao() -> pd.DataFrame:
    """Junta as listas de exemplo em uma tabela (DataFrame) com texto e rótulo."""
    return pd.DataFrame(
        {
            "texto": SPAM + NAO_SPAM,
            "rotulo": [1] * len(SPAM) + [0] * len(NAO_SPAM),
        }
    )


# ---------------------------------------------------------------------------
# 3) TREINO DO MODELO
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Treinando o modelo, aguarde alguns segundos...")
def treinar_modelo(textos: tuple, rotulos: tuple):
    """
    Treina a rede neural e devolve (modelo, acurácia no conjunto de teste).

    @st.cache_resource faz o Streamlit lembrar do resultado: o modelo só é
    treinado de novo se os dados (textos/rótulos) mudarem. Por isso os
    parâmetros são tuplas: elas são "hasheáveis", ou seja, o Streamlit
    consegue comparar se mudaram.
    """
    keras.utils.set_random_seed(SEED)

    x = np.array(textos, dtype=object)  # aqui só usamos o NumPy para embaralhar/indexar
    y = np.array(rotulos, dtype="float32")

    # Embaralha e separa 80% para treino e 20% para teste
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(x))
    corte = int(len(x) * 0.8)
    treino, teste = indices[:corte], indices[corte:]

    # O Keras 3 não aceita arrays de texto do NumPy. Solução: converter para
    # tensores do TensorFlow (tf.constant), que têm o tipo tf.string.
    # Formato (n, 1): uma coluna com um texto por linha.
    def para_tensor(idx):
        return tf.reshape(tf.constant([str(t) for t in x[idx]]), (-1, 1))

    x_treino, x_teste = para_tensor(treino), para_tensor(teste)
    ds_treino = tf.data.Dataset.from_tensor_slices((x_treino, y[treino])).batch(8)
    ds_teste = tf.data.Dataset.from_tensor_slices((x_teste, y[teste])).batch(8)

    # Camada que transforma texto em números (TF-IDF):
    # palavras raras e marcantes pesam mais que palavras comuns.
    vetorizador = keras.layers.TextVectorization(
        max_tokens=MAX_PALAVRAS,
        output_mode="tf_idf",
        standardize="lower_and_strip_punctuation",
    )
    # adapt() aprende o vocabulário usando APENAS os dados de treino
    vetorizador.adapt(tf.data.Dataset.from_tensor_slices(x_treino[:, 0]).batch(16))

    # Modelo: texto -> vetor TF-IDF -> camadas densas -> probabilidade (0 a 1)
    entrada = keras.Input(shape=(1,), dtype=tf.string)
    camada = vetorizador(entrada)
    camada = keras.layers.Dense(32, activation="relu")(camada)
    camada = keras.layers.Dropout(0.4)(camada)  # evita "decorar" os exemplos
    saida = keras.layers.Dense(1, activation="sigmoid")(camada)
    modelo = keras.Model(entrada, saida)

    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    modelo.fit(ds_treino, epochs=EPOCAS, verbose=0)

    # Avalia em mensagens que o modelo nunca viu durante o treino
    _, acuracia = modelo.evaluate(ds_teste, verbose=0)
    return modelo, float(acuracia)


def prever_spam(modelo, mensagem: str) -> float:
    """Devolve a probabilidade (0 a 1) de a mensagem ser spam."""
    entrada = tf.constant([[mensagem]])  # tensor tf.string de formato (1, 1)
    return float(modelo(entrada, training=False).numpy()[0][0])


# ---------------------------------------------------------------------------
# 4) LEITURA DO CSV DA EMPRESA (OPCIONAL)
# ---------------------------------------------------------------------------
def ler_csv(arquivo) -> pd.DataFrame:
    """
    Lê um CSV com as colunas 'texto' e 'rotulo'.
    O rótulo pode ser: spam/ham/nao_spam ou 1/0.
    """
    df = pd.read_csv(arquivo)
    df.columns = [c.strip().lower() for c in df.columns]

    if not {"texto", "rotulo"}.issubset(df.columns):
        raise ValueError("O CSV precisa ter as colunas 'texto' e 'rotulo'.")

    mapa = {"spam": 1, "1": 1, "ham": 0, "nao_spam": 0, "nao spam": 0, "0": 0}
    df["rotulo"] = df["rotulo"].astype(str).str.strip().str.lower().map(mapa)
    df = df.dropna(subset=["texto", "rotulo"])
    df["rotulo"] = df["rotulo"].astype(int)

    if df["rotulo"].nunique() < 2 or len(df) < 20:
        raise ValueError(
            "O CSV precisa ter ao menos 20 linhas, com exemplos de spam e de não spam."
        )
    return df[["texto", "rotulo"]]


# ---------------------------------------------------------------------------
# 5) INTERFACE (STREAMLIT)
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Detector de Spam",
        page_icon="📧",
        layout="centered",
    )

    # ----- Barra lateral: configurações -----
    with st.sidebar:
        st.header("Configurações")
        limiar = st.slider(
            "Sensibilidade (limiar de spam)",
            min_value=0.10,
            max_value=0.90,
            value=LIMIAR_PADRAO,
            step=0.05,
            help="Quanto menor o valor, mais mensagens serão marcadas como spam.",
        )

        st.divider()
        st.subheader("Treinar com dados da empresa")
        arquivo = st.file_uploader(
            "CSV com as colunas 'texto' e 'rotulo' (spam/nao_spam ou 1/0)",
            type="csv",
        )

    # ----- Define quais dados usar no treino -----
    df = montar_dataframe_padrao()
    origem = "base de exemplo embutida"
    if arquivo is not None:
        try:
            df = ler_csv(arquivo)
            origem = f"CSV enviado ({len(df)} mensagens)"
        except Exception as erro:  # mostra o motivo em vez de quebrar o app
            st.sidebar.error(f"Não foi possível usar o CSV: {erro}")

    modelo, acuracia = treinar_modelo(
        tuple(df["texto"].astype(str)), tuple(df["rotulo"].astype(int))
    )

    with st.sidebar:
        st.divider()
        st.caption(f"Base de treino: {origem}")
        st.caption(f"Acurácia no teste: {acuracia:.0%}")

    # ----- Área principal -----
    st.title("📧 Detector de Spam")
    st.write(
        "Cole o conteúdo do e-mail abaixo para verificar se a mensagem "
        "é spam ou não."
    )

    mensagem = st.text_area(
        "Mensagem do e-mail",
        height=220,
        placeholder="Cole aqui o assunto e o corpo do e-mail...",
    )

    if st.button("Analisar mensagem", type="primary"):
        if not mensagem.strip():
            st.warning("Digite ou cole uma mensagem para analisar.")
        else:
            with st.spinner("Analisando..."):
                prob = prever_spam(modelo, mensagem)

            col1, col2 = st.columns(2)
            col1.metric("Probabilidade de spam", f"{prob:.1%}")
            col2.metric("Probabilidade de não spam", f"{1 - prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))

            if prob >= limiar:
                st.error("🚫 **Classificação: SPAM**. Não recomendamos enviar ou confiar neste conteúdo.")
            else:
                st.success("✅ **Classificação: NÃO É SPAM**. O conteúdo parece legítimo.")

    st.divider()
    st.caption(
        "O resultado é uma estimativa de um modelo de IA e deve apoiar, "
        "não substituir, a revisão humana."
    )


if __name__ == "__main__":
    main()