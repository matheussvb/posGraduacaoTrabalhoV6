import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.config import SEMENTE
from src.visualizacao import salvar_figura


def importancia_por_permutacao(modelo, x, y):
    resultado = permutation_importance(modelo, x, y, scoring="f1", n_repeats=20, random_state=SEMENTE)
    importancia = pd.DataFrame(
        {"media": resultado.importances_mean, "desvio": resultado.importances_std},
        index=x.columns,
    )
    return importancia.sort_values("media", ascending=False)


def grafico_importancia(importancia, nome_modelo, quantidade=15):
    principais = importancia.head(quantidade).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(principais.index, principais["media"], xerr=principais["desvio"], color="#4C72B0")
    ax.set_title(f"Importância por permutação - {nome_modelo}")
    ax.set_xlabel("Queda média do F1 ao embaralhar a variável")
    return salvar_figura(fig, "importancia_variaveis.png")


def calcular_shap(modelo, x_referencia, x_explicar):
    colunas = list(x_explicar.columns)

    def probabilidade_maligno(dados):
        return modelo.predict_proba(pd.DataFrame(dados, columns=colunas))[:, 1]

    amostra = x_referencia.sample(min(100, len(x_referencia)), random_state=SEMENTE)
    mascara = shap.maskers.Independent(amostra, max_samples=len(amostra))
    explicador = shap.Explainer(probabilidade_maligno, mascara, feature_names=colunas, seed=SEMENTE)
    return explicador(x_explicar, silent=True)


def ranking_shap(valores_shap):
    media_absoluta = abs(valores_shap.values).mean(axis=0)
    return pd.Series(media_absoluta, index=valores_shap.feature_names).sort_values(ascending=False)


def grafico_shap_resumo(valores_shap):
    shap.plots.beeswarm(valores_shap, max_display=15, show=False)
    plt.xlabel("Valor SHAP (efeito na probabilidade de maligno)")
    fig = plt.gcf()
    fig.suptitle("SHAP - impacto de cada variável na probabilidade de maligno")
    return salvar_figura(fig, "shap_resumo.png")


def grafico_shap_barras(valores_shap):
    shap.plots.bar(valores_shap, max_display=15, show=False)
    plt.xlabel("Média do valor SHAP absoluto")
    fig = plt.gcf()
    fig.suptitle("SHAP - importância média absoluta")
    return salvar_figura(fig, "shap_barras.png")


def grafico_shap_paciente(valores_shap, posicao):
    shap.plots.waterfall(valores_shap[posicao], max_display=12, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP - explicação de um exame do conjunto de teste")
    return salvar_figura(fig, "shap_exemplo_paciente.png")
