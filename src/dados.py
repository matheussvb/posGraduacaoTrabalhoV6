import pandas as pd

from src.config import ARQUIVO_DADOS, COLUNA_ALVO


def carregar_dados(caminho=ARQUIVO_DADOS):
    df = pd.read_csv(caminho)
    df.columns = [coluna.strip().replace(" ", "_") for coluna in df.columns]
    colunas_inuteis = [coluna for coluna in df.columns if coluna == "id" or coluna.startswith("Unnamed")]
    return df.drop(columns=colunas_inuteis)


def separar_x_y(df):
    x = df.drop(columns=[COLUNA_ALVO])
    y = df[COLUNA_ALVO].map({"M": 1, "B": 0}).astype(int)
    return x, y


def resumo_dos_dados(df):
    contagem = df[COLUNA_ALVO].value_counts()
    return {
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "valores_ausentes": int(df.isna().sum().sum()),
        "linhas_duplicadas": int(df.duplicated().sum()),
        "malignos": int(contagem.get("M", 0)),
        "benignos": int(contagem.get("B", 0)),
    }
