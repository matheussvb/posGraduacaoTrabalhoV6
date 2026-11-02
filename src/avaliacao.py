from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def calcular_metricas(y_real, y_previsto):
    return {
        "accuracy": round(float(accuracy_score(y_real, y_previsto)), 4),
        "precision": round(float(precision_score(y_real, y_previsto, zero_division=0)), 4),
        "recall": round(float(recall_score(y_real, y_previsto, zero_division=0)), 4),
        "f1": round(float(f1_score(y_real, y_previsto, zero_division=0)), 4),
        "matriz_confusao": confusion_matrix(y_real, y_previsto, labels=[0, 1]).tolist(),
    }


def escolher_melhor_modelo(metricas_por_modelo):
    def criterio(nome):
        metricas = metricas_por_modelo[nome]
        return metricas["recall"], metricas.get("recall_cv_treino", 0), metricas["f1"]

    return max(metricas_por_modelo, key=criterio)
