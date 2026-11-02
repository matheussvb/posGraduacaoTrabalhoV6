import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import COLUNA_ALVO, PROPORCAO_TESTE, PROPORCAO_VALIDACAO, SEMENTE


def limpar_dados(df):
    df = df.drop_duplicates().copy()
    df[COLUNA_ALVO] = df[COLUNA_ALVO].astype(str).str.strip().str.upper()
    df = df[df[COLUNA_ALVO].isin(["M", "B"])].copy()

    colunas_numericas = df.columns.drop(COLUNA_ALVO)
    df[colunas_numericas] = df[colunas_numericas].apply(pd.to_numeric, errors="coerce")
    df[colunas_numericas] = df[colunas_numericas].mask(df[colunas_numericas] < 0)
    return df.reset_index(drop=True)


def dividir_dados(x, y):
    
    x_resto, x_teste, y_resto, y_teste = train_test_split(
        x, y, test_size=PROPORCAO_TESTE, stratify=y, random_state=SEMENTE
    )
    
    proporcao_validacao_no_resto = PROPORCAO_VALIDACAO / (1 - PROPORCAO_TESTE)
    
    x_treino, x_validacao, y_treino, y_validacao = train_test_split(
        x_resto, y_resto, test_size=proporcao_validacao_no_resto, stratify=y_resto, random_state=SEMENTE
    )
    return {
        "x_treino": x_treino,
        "y_treino": y_treino,
        "x_validacao": x_validacao,
        "y_validacao": y_validacao,
        "x_teste": x_teste,
        "y_teste": y_teste,
    }


def criar_pipeline(modelo):
    return Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalonador", StandardScaler()),
        ("modelo", modelo),
    ])
