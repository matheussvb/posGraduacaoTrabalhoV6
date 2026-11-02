import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import ARQUIVO_ESTATISTICAS, COLUNA_ALVO, PASTA_IMAGENS

NOMES_CLASSES = {"B": "Benigno", "M": "Maligno"}

CORES_CLASSES = {"Benigno": "#4C72B0", "Maligno": "#C44E52"}

VARIAVEIS_PRINCIPAIS = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concave_points_mean",
]


def salvar_figura(fig, nome_arquivo):
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_IMAGENS / nome_arquivo, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig


def estatisticas_descritivas(df):
    tabela = df.drop(columns=[COLUNA_ALVO]).describe().T.round(4)
    ARQUIVO_ESTATISTICAS.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(ARQUIVO_ESTATISTICAS)
    return tabela


def media_por_classe(df):
    medias = df.groupby(COLUNA_ALVO)[VARIAVEIS_PRINCIPAIS].mean().T
    medias = medias.rename(columns=NOMES_CLASSES)
    medias["razao_maligno_benigno"] = medias["Maligno"] / medias["Benigno"]
    return medias.round(3)


def grafico_balanco_classes(df):
    contagem = df[COLUNA_ALVO].map(NOMES_CLASSES).value_counts().reindex(["Benigno", "Maligno"])
    total = contagem.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(contagem.index, contagem.values, color=[CORES_CLASSES[c] for c in contagem.index])
    ax.bar_label(barras, labels=[f"{valor} ({valor / total:.1%})" for valor in contagem.values])
    ax.set_title("Balanço das classes")
    ax.set_ylabel("Quantidade de exames")
    ax.set_ylim(0, contagem.max() * 1.15)
    return salvar_figura(fig, "balanco_classes.png")


def grafico_distribuicoes(df):
    dados = df.copy()
    dados["Diagnóstico"] = dados[COLUNA_ALVO].map(NOMES_CLASSES)

    fig, eixos = plt.subplots(2, 3, figsize=(15, 8))
    for coluna, ax in zip(VARIAVEIS_PRINCIPAIS, eixos.flatten()):
        sns.histplot(data=dados, x=coluna, hue="Diagnóstico", kde=True, palette=CORES_CLASSES, ax=ax)
        ax.set_title(coluna)
        ax.set_xlabel("")
        ax.set_ylabel("Quantidade")
    fig.suptitle("Distribuição das principais variáveis por diagnóstico")
    fig.tight_layout()
    return salvar_figura(fig, "distribuicoes.png")


def grafico_boxplots(df):
    dados = df.copy()
    dados["Diagnóstico"] = dados[COLUNA_ALVO].map(NOMES_CLASSES)

    fig, eixos = plt.subplots(2, 3, figsize=(15, 8))
    for coluna, ax in zip(VARIAVEIS_PRINCIPAIS, eixos.flatten()):
        sns.boxplot(data=dados, x="Diagnóstico", y=coluna, hue="Diagnóstico", palette=CORES_CLASSES, ax=ax)
        ax.set_title(coluna)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Boxplots das principais variáveis por diagnóstico")
    fig.tight_layout()
    return salvar_figura(fig, "boxplots.png")


def grafico_correlacao(df):
    correlacao = df.drop(columns=[COLUNA_ALVO]).corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlacao, cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Matriz de correlação entre as variáveis")
    return salvar_figura(fig, "correlacao.png")


def correlacao_com_alvo(df):
    alvo = df[COLUNA_ALVO].map({"M": 1, "B": 0})
    return df.drop(columns=[COLUNA_ALVO]).corrwith(alvo).sort_values()


def grafico_correlacao_com_alvo(df):
    correlacao = correlacao_com_alvo(df)

    fig, ax = plt.subplots(figsize=(8, 10))
    cores = ["#C44E52" if valor > 0 else "#4C72B0" for valor in correlacao.values]
    ax.barh(correlacao.index, correlacao.values, color=cores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlação de cada variável com o diagnóstico (1 = maligno)")
    ax.set_xlabel("Correlação de Pearson")
    return salvar_figura(fig, "correlacao_com_alvo.png")


def grafico_matrizes_confusao(matrizes, titulo, nome_arquivo):
    fig, eixos = plt.subplots(1, len(matrizes), figsize=(5 * len(matrizes), 4), squeeze=False)
    for (nome, matriz), ax in zip(matrizes.items(), eixos[0]):
        sns.heatmap(
            matriz,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benigno", "Maligno"],
            yticklabels=["Benigno", "Maligno"],
            ax=ax,
        )
        ax.set_title(nome)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
    fig.suptitle(titulo)
    fig.tight_layout()
    return salvar_figura(fig, nome_arquivo)


def grafico_comparacao_modelos(metricas_por_modelo):
    tabela = pd.DataFrame(metricas_por_modelo).T[["accuracy", "precision", "recall", "f1"]].astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    tabela.plot.bar(ax=ax, rot=0)
    ax.set_title("Comparação dos modelos no conjunto de validação")
    ax.set_ylabel("Valor da métrica")
    ax.set_ylim(min(0.8, tabela.values.min() - 0.05), 1.02)
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    return salvar_figura(fig, "comparacao_modelos.png")
