"""Treinamento e avaliação TF-IDF + XGBoost do SaudeIA."""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from services.datasetService import dataset_completo
from services.vetorizacaoService import criar_vetorizador, criar_encoderY


def treinar_modelo():
    """Devolve modelo, tfidf, encoder, X_teste e Y_teste."""
    df = dataset_completo()
    textos = df["sintomas"]
    encoder, Y = criar_encoderY(df["diagnostico"])

    # Divisão ANTES de ajustar o TF-IDF: evita vazamento de vocabulário.
    contagens = df["diagnostico"].value_counts()
    if contagens.min() < 2:
        raise ValueError("Cada classe precisa ter pelo menos dois registros")
    X_treino_txt, X_teste_txt, Y_treino, Y_teste = train_test_split(
        textos, Y, test_size=0.30, random_state=42, stratify=Y
    )

    # fit_transform apenas no treino; transform apenas no teste.
    tfidf, X_treino, X_teste = criar_vetorizador(
        X_treino_txt, X_teste_txt
    )

    modelo = XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=2,
    )
    modelo.fit(X_treino, Y_treino)
    return modelo, tfidf, encoder, X_teste, Y_teste


def acuracia_modelo():
    """Calcula a porcentagem de acertos nos registros reservados."""
    modelo, _, _, X_teste, Y_teste = treinar_modelo()
    Y_previsto = modelo.predict(X_teste)
    return accuracy_score(Y_teste, Y_previsto) * 100


def main():
    modelo, tfidf, encoder, X_teste, Y_teste = treinar_modelo()
    Y_previsto = modelo.predict(X_teste)
    print("=== SAUDEIA — TREINAMENTO TF-IDF + XGBOOST ===")
    print("Registros de teste:", len(Y_teste))
    print("Classes:", len(encoder.classes_))
    print("Termos do vocabulário:", len(tfidf.get_feature_names_out()))
    print("Acurácia:", round(accuracy_score(Y_teste, Y_previsto) * 100, 2), "%")
    print(classification_report(
        Y_teste, Y_previsto,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_, zero_division=0,
    ))


if __name__ == "__main__":
    main()
