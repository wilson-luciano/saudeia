"""Treinamento e avaliação TF-IDF + XGBoost do SaudeIA."""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from services.datasetService import dataset_completo
from services.vetorizacaoService import criar_vetorizador, criar_encoderY


def treinar_modelo():
    """Treina o modelo e devolve os componentes e os dados de teste."""

    # Carrega o conjunto de dados do projeto.
    df = dataset_completo()

    textos = df["sintomas"]
    diagnosticos = df["diagnostico"]

    # Verifica se há exemplos suficientes para a divisão estratificada.
    contagens = diagnosticos.value_counts()

    if contagens.min() < 2:
        raise ValueError(
            "Cada classe precisa ter pelo menos dois registros"
        )

    # Transforma as classes em números.
    encoder, y = criar_encoderY(diagnosticos)

    # Divide os textos ANTES do ajuste do TF-IDF.
    X_treino_txt, X_teste_txt, y_treino, y_teste = train_test_split(
        textos,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # Aprende o vocabulário somente com os dados de treinamento.
    tfidf, X_treino, X_teste = criar_vetorizador(
        X_treino_txt,
        X_teste_txt,
    )

    # Configura o classificador XGBoost.
    modelo = XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=2,
    )

    # Treina o modelo.
    modelo.fit(X_treino, y_treino)

    return modelo, tfidf, encoder, X_teste, y_teste


def acuracia_modelo():
    """Retorna a acurácia percentual do modelo."""

    modelo, _, _, X_teste, y_teste = treinar_modelo()

    y_previsto = modelo.predict(X_teste)

    return accuracy_score(y_teste, y_previsto) * 100


def main():
    """Executa o treinamento e apresenta as métricas."""

    modelo, tfidf, encoder, X_teste, y_teste = treinar_modelo()

    y_previsto = modelo.predict(X_teste)

    print("=== SAUDEIA - TREINAMENTO TF-IDF + XGBOOST ===")
    print("Registros de teste:", len(y_teste))
    print("Classes:", len(encoder.classes_))
    print("Termos do vocabulário:", len(tfidf.get_feature_names_out()))

    print(
        "Acurácia:",
        round(accuracy_score(y_teste, y_previsto) * 100, 2),
        "%",
    )

    print(
        classification_report(
            y_teste,
            y_previsto,
            labels=list(range(len(encoder.classes_))),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
