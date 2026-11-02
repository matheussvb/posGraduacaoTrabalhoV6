from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.config import SEMENTE
from src.preprocessamento import criar_pipeline


def criar_modelos():
    return {
        "Regressão Logística": criar_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEMENTE)
        ),
        "KNN": criar_pipeline(KNeighborsClassifier(n_neighbors=7)),
        "Random Forest": criar_pipeline(
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEMENTE)
        ),
    }
