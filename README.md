# Tech Challenge - Fase 1: diagnóstico de câncer de mama com Machine Learning

Projeto da Fase 1 do Tech Challenge da pós-graduação. A ideia é a base de um sistema de apoio ao diagnóstico para um hospital universitário: a partir das medidas de um exame de punção aspirativa por agulha fina (PAAF) de um nódulo de mama, o modelo classifica o tumor como **maligno** ou **benigno**.

O modelo é uma ferramenta de **apoio à triagem**. A decisão final é sempre do médico.

## Dataset

- **Breast Cancer Wisconsin (Diagnostic)**: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Cópia usada no projeto: `data/breast_cancer_wisconsin.csv`
- 569 exames, 30 variáveis numéricas e o diagnóstico (`M` = maligno, `B` = benigno)
- 357 benignos (62,7%) e 212 malignos (37,3%)

As 30 variáveis vêm de 10 características do núcleo das células (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal), cada uma medida como média (`_mean`), erro padrão (`_se`) e pior valor (`_worst`).

## Resultados principais

Três modelos foram comparados no conjunto de validação. A métrica principal é o **recall da classe maligna**, porque deixar passar um câncer (falso negativo) é muito mais grave do que pedir um exame a mais (falso positivo).

| Modelo (validação) | Accuracy | Precision | Recall | F1 | Recall CV no treino |
| --- | --- | --- | --- | --- | --- |
| Regressão Logística | 0,965 | 0,939 | 0,969 | 0,954 | 0,953 |
| KNN | 0,977 | 0,969 | 0,969 | 0,969 | 0,885 |
| Random Forest | 0,965 | 0,968 | 0,938 | 0,952 | 0,939 |

Regressão Logística e KNN empataram no recall da validação. O desempate é o recall médio da validação cruzada (5 partes) no treino, e por ele a **Regressão Logística** foi escolhida. Ela foi avaliada uma única vez no teste:

| Modelo final (teste) | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Regressão Logística | 0,988 | 1,000 | 0,969 | 0,984 |

No teste o modelo encontrou 31 dos 32 tumores malignos, com 1 falso negativo e nenhum falso positivo. Pela importância por permutação e pelo SHAP, as variáveis que mais pesam na decisão são `texture_worst`, `radius_se`, `radius_worst`, `concave_points_mean` e `concave_points_worst`. Em geral são medidas de tamanho e de irregularidade do núcleo. A textura sozinha separa pouco as classes, mas ajuda quando combinada com as outras.

Todos os gráficos estão em `reports/figuras/` e a análise completa está no relatório técnico `reports/relatorio_tecnico.pdf`.

## Estrutura do projeto

```
posGraduacaoTrabalhoV6/
├── data/
│   └── breast_cancer_wisconsin.csv   dataset
├── notebooks/
│   └── demonstracao.ipynb            demonstração passo a passo, com as saídas
├── reports/
│   ├── figuras/                      gráficos gerados pelo pipeline
│   ├── estatisticas_descritivas.csv
│   ├── metricas.json                 métricas de validação e teste
│   └── relatorio_tecnico.pdf         relatório técnico
├── src/
│   ├── config.py                     caminhos, semente e proporções
│   ├── dados.py                      carga do CSV e separação de X e y
│   ├── preprocessamento.py           limpeza, divisão treino/validação/teste e pipeline
│   ├── modelos.py                    os três modelos
│   ├── avaliacao.py                  métricas e escolha do melhor modelo
│   ├── visualizacao.py               estatísticas e gráficos
│   ├── interpretacao.py              importância por permutação e SHAP
│   └── treinar.py                    ponto de entrada do pipeline
├── Dockerfile
└── requirements.txt
```

## Como funciona o pipeline

1. **Carga**: lê o CSV e remove colunas sem informação (`id` e a coluna vazia do arquivo original).
2. **Limpeza**: remove duplicadas, padroniza o diagnóstico, descarta diagnósticos inválidos e trata valores negativos ou não numéricos como ausentes.
3. **Exploração**: estatísticas descritivas, balanço das classes, distribuições, boxplots e correlação.
4. **Divisão**: 70% treino, 15% validação e 15% teste, estratificada e com semente fixa (42).
5. **Pré-processamento**: `Pipeline` do scikit-learn com imputação pela mediana e `StandardScaler`, ajustado só no treino.
6. **Modelos**: Regressão Logística, KNN e Random Forest treinados no treino e comparados na validação.
7. **Teste**: o modelo escolhido é avaliado uma única vez no teste.
8. **Interpretação**: importância por permutação e SHAP.
9. **Saídas**: métricas em JSON, gráficos em PNG.

## Como rodar

**Pré-requisitos** (instalar uma vez): [Git](https://git-scm.com/downloads) e [Python 3.13](https://www.python.org/downloads/). No Windows, marque *Add python.exe to PATH* no instalador. Para usar Docker em vez de Python, instale só o [Docker Desktop](https://www.docker.com/products/docker-desktop/).

1. Baixe o projeto (o dataset já vem junto em `data/`, não precisa baixar nada do Kaggle):

   ```bash
   git clone https://github.com/matheussvb/posGraduacaoTrabalhoV6.git
   cd posGraduacaoTrabalhoV6
   ```

2. Crie e ative o ambiente virtual:

   ```bash
   python -m venv .venv

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   # Linux ou macOS
   source .venv/bin/activate
   ```

   No PowerShell, se aparecer erro de execução de scripts, rode antes `Set-ExecutionPolicy -Scope Process Bypass`.

3. Instale as dependências e rode o pipeline:

   ```bash
   python -m pip install -r requirements.txt
   python -m src.treinar
   ```

4. Confira o resultado: o terminal mostra as métricas e a pasta `reports/` recebe `metricas.json`, `estatisticas_descritivas.csv` e `figuras/`.

Alternativa sem instalar Python: siga a seção [Como rodar com Docker](#como-rodar-com-docker) depois do `git clone`.

## Notebook

Para abrir e executar a demonstração:

```bash
python -m jupyter notebook notebooks/demonstracao.ipynb
```

Ou para executar o notebook inteiro pelo terminal:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/demonstracao.ipynb
```

## Como rodar com Docker

Construir a imagem:

```bash
docker build -t cancer-de-mama .
```

Rodar o pipeline e trazer os resultados para a pasta `reports/` da máquina:

```bash
# Linux, macOS ou Git Bash
docker run --rm -v "$(pwd)/reports:/app/reports" cancer-de-mama

# Windows (PowerShell)
docker run --rm -v "${PWD}/reports:/app/reports" cancer-de-mama
```

## Relatório e vídeo

- Relatório técnico: `reports/relatorio_tecnico.pdf`
- Vídeo de demonstração: https://youtu.be/zjzghIY_BbQ
- Repositório: https://github.com/matheussvb/posGraduacaoTrabalhoV6

## Autores

- Matheus Silva Vilas Boas
- Tatiane Pereira Rocha
- Cyro Pereira Silva Souza
