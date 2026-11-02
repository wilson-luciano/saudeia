"""Transformação de textos e codificação das classes do SaudeIA."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from services.datasetService import dataset_completo


def criar_vetorizador(textos_treino, textos_teste=None):
    """Aprende TF-IDF somente no treino e transforma o teste sem reajuste."""
    tfidf = TfidfVectorizer()
    X_treino = tfidf.fit_transform(textos_treino)
    if textos_teste is None:
        return tfidf, X_treino
    X_teste = tfidf.transform(textos_teste)
    return tfidf, X_treino, X_teste


def criar_encoderY(rotulos=None):
    """Codifica os nomes das classes; aceita os rótulos fornecidos."""
    if rotulos is None:
        rotulos = dataset_completo()["diagnostico"]
    encoder = LabelEncoder()
    Y = encoder.fit_transform(rotulos)
    return encoder, Y


def main():
    # Demonstra o serviço sem usar os dados de teste para ajustar o TF-IDF.
    from sklearn.model_selection import train_test_split
    df = dataset_completo()
    treino, teste = train_test_split(
        df["sintomas"], test_size=0.30, random_state=42
    )
    tfidf, X_treino, X_teste = criar_vetorizador(treino, teste)
    encoder, Y = criar_encoderY(df["diagnostico"])
    print("Matriz de treino:", X_treino.shape)
    print("Matriz de teste:", X_teste.shape)
    print("Classes:", len(encoder.classes_))
    print("Rótulos codificados:", len(Y))


if __name__ == "__main__":
    main()
