from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

PASTA_DE_DADOS = RAIZ / "data"

ARQUIVO_DADOS = PASTA_DE_DADOS / "breast_cancer_wisconsin.csv"

PASTA_RESULTADOS = RAIZ / "reports"

PASTA_IMAGENS = PASTA_RESULTADOS / "figuras"

SEMENTE = 42


PROPORCAO_TREINO = 0.70
PROPORCAO_VALIDACAO = 0.15
PROPORCAO_TESTE = 0.15

COLUNA_ALVO = "diagnosis"

ARQUIVO_METRICAS = PASTA_RESULTADOS / "metricas.json"

ARQUIVO_ESTATISTICAS = PASTA_RESULTADOS / "estatisticas_descritivas.csv"
