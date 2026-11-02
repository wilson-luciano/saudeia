"""Vetorização TF-IDF e codificação dos diagnósticos do SaudeIA."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def criar_vetorizador(textos_treino, textos_teste):
    """
    Ajusta o TF-IDF apenas nos textos de treinamento.

    Retorna:
        tfidf: vetorizador ajustado;
        X_treino: matriz dos textos de treinamento;
        X_teste: matriz dos textos de teste.
    """
    tfidf = TfidfVectorizer()

    # Aprende o vocabulário exclusivamente com o treinamento.
    X_treino = tfidf.fit_transform(textos_treino)

    # Reutiliza o vocabulário sem aprender palavras novas.
    X_teste = tfidf.transform(textos_teste)

    return tfidf, X_treino, X_teste


def criar_encoderY(diagnosticos):
    """
    Converte os diagnósticos em rótulos numéricos.

    Retorna o codificador ajustado e os rótulos codificados.
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(diagnosticos)

    return encoder, y
