"""Testes rápidos dos serviços de dados e vetorização."""
import pandas as pd
from services.vetorizacaoService import criar_vetorizador, criar_encoderY


def test_tfidf_nao_aprende_palavra_do_teste():
    # 'exclusivadoteste' aparece somente no teste e não pode entrar no vocabulário.
    tfidf, X_treino, X_teste = criar_vetorizador(
        ["febre persistente", "dor muscular"],
        ["exclusivadoteste febre"],
    )
    assert "exclusivadoteste" not in tfidf.vocabulary_
    assert X_treino.shape[1] == X_teste.shape[1]


def test_encoder_devolve_classes_numericas():
    encoder, y = criar_encoderY(pd.Series(["classe_b", "classe_a", "classe_b"]))
    assert len(encoder.classes_) == 2
    assert len(y) == 3


def test_vetorizador_nao_usa_fit_no_teste():
    tfidf, _, _ = criar_vetorizador(["palavra treino"], ["outrapalavra teste"])
    assert "outrapalavra" not in tfidf.get_feature_names_out()
