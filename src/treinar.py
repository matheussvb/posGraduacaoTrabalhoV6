import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import interpretacao, visualizacao
from src.avaliacao import calcular_metricas, escolher_melhor_modelo
from src.config import ARQUIVO_METRICAS, PASTA_RESULTADOS, SEMENTE
from src.dados import carregar_dados, resumo_dos_dados, separar_x_y
from src.modelos import criar_modelos
from src.preprocessamento import dividir_dados, limpar_dados


def gerar_graficos_exploratorios(df):
    visualizacao.grafico_balanco_classes(df)
    visualizacao.grafico_distribuicoes(df)
    visualizacao.grafico_boxplots(df)
    visualizacao.grafico_correlacao(df)
    visualizacao.grafico_correlacao_com_alvo(df)


def treinar_e_comparar(divisao):
    modelos = criar_modelos()
    metricas_validacao = {}
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)

    for nome, modelo in modelos.items():
        recall_cv = cross_val_score(modelo, divisao["x_treino"], divisao["y_treino"], cv=folds, scoring="recall")
        modelo.fit(divisao["x_treino"], divisao["y_treino"])
        previsto = modelo.predict(divisao["x_validacao"])
        metricas_validacao[nome] = calcular_metricas(divisao["y_validacao"], previsto)
        metricas_validacao[nome]["recall_cv_treino"] = round(float(recall_cv.mean()), 4)

    return modelos, metricas_validacao


def avaliar_no_teste(modelo, divisao):
    previsto = modelo.predict(divisao["x_teste"])
    return calcular_metricas(divisao["y_teste"], previsto)


def interpretar_modelo(modelo, nome_modelo, divisao):
    importancia = interpretacao.importancia_por_permutacao(modelo, divisao["x_teste"], divisao["y_teste"])
    interpretacao.grafico_importancia(importancia, nome_modelo)

    valores_shap = interpretacao.calcular_shap(modelo, divisao["x_treino"], divisao["x_teste"])
    interpretacao.grafico_shap_resumo(valores_shap)
    interpretacao.grafico_shap_barras(valores_shap)

    posicao_maligno = list(divisao["y_teste"]).index(1)
    interpretacao.grafico_shap_paciente(valores_shap, posicao_maligno)

    return {
        "importancia_permutacao": importancia["media"].head(10).round(4).to_dict(),
        "shap_media_absoluta": interpretacao.ranking_shap(valores_shap).head(10).round(4).to_dict(),
    }


def mostrar_tabela(titulo, metricas_por_modelo):
    print(f"\n{titulo}")
    tabela = pd.DataFrame(metricas_por_modelo).T.drop(columns=["matriz_confusao"])
    print(tabela.to_string())


def main():
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    df = carregar_dados()
    resumo_original = resumo_dos_dados(df)
    df = limpar_dados(df)
    resumo_limpo = resumo_dos_dados(df)
    print("Dados originais:", resumo_original)
    print("Dados após limpeza:", resumo_limpo)

    estatisticas = visualizacao.estatisticas_descritivas(df)
    medias = visualizacao.media_por_classe(df)
    correlacao_alvo = visualizacao.correlacao_com_alvo(df)
    gerar_graficos_exploratorios(df)

    x, y = separar_x_y(df)
    divisao = dividir_dados(x, y)
    tamanhos = {
        "treino": len(divisao["x_treino"]),
        "validacao": len(divisao["x_validacao"]),
        "teste": len(divisao["x_teste"]),
    }
    print("Tamanho dos conjuntos:", tamanhos)

    modelos, metricas_validacao = treinar_e_comparar(divisao)
    mostrar_tabela("Métricas na validação", metricas_validacao)
    visualizacao.grafico_comparacao_modelos(metricas_validacao)
    visualizacao.grafico_matrizes_confusao(
        {nome: metricas["matriz_confusao"] for nome, metricas in metricas_validacao.items()},
        "Matrizes de confusão na validação",
        "matrizes_confusao_validacao.png",
    )

    nome_melhor = escolher_melhor_modelo(metricas_validacao)
    modelo_final = modelos[nome_melhor]
    print(f"\nModelo escolhido pela validação: {nome_melhor}")

    metricas_teste = avaliar_no_teste(modelo_final, divisao)
    mostrar_tabela("Métricas no teste (avaliação única)", {nome_melhor: metricas_teste})
    visualizacao.grafico_matrizes_confusao(
        {nome_melhor: metricas_teste["matriz_confusao"]},
        "Matriz de confusão no teste",
        "matriz_confusao_teste.png",
    )

    print("\nCalculando importância das variáveis e SHAP...")
    resultado_interpretacao = interpretar_modelo(modelo_final, nome_melhor, divisao)

    resultado = {
        "modelo_escolhido": nome_melhor,
        "criterio_de_escolha": "maior recall na validação; empate decidido pelo recall da validação cruzada no treino e depois pelo F1",
        "semente": SEMENTE,
        "dados_originais": resumo_original,
        "dados_limpos": resumo_limpo,
        "tamanho_dos_conjuntos": tamanhos,
        "validacao": metricas_validacao,
        "teste": metricas_teste,
        "interpretacao": resultado_interpretacao,
        "exploracao": {
            "estatisticas_principais": estatisticas.loc[visualizacao.VARIAVEIS_PRINCIPAIS, ["mean", "std", "min", "max"]].to_dict(orient="index"),
            "media_por_classe": medias.to_dict(orient="index"),
            "correlacao_com_alvo": correlacao_alvo.sort_values(ascending=False).round(4).to_dict(),
        },
    }

    with open(ARQUIVO_METRICAS, "w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2)

    print(f"\nArquivos gerados em {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()
