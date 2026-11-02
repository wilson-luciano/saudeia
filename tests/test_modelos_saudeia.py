"""Testes rápidos dos arquivos e da inferência do SaudeIA."""

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from services.diagnosticoIA import DiagnosticoIA


def preparar_modelos_teste(pasta):
    """Cria componentes mínimos em uma pasta temporária."""
    textos = [
        "febre tosse", "tosse cansaço", "febre dor",
        "ruido bomba", "vibracao motor", "falha rolamento",
    ]
    rotulos = ["saude", "saude", "saude", "maquina", "maquina", "maquina"]

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(textos)
    encoder = LabelEncoder()
    y = encoder.fit_transform(rotulos)

    modelo = XGBClassifier(n_estimators=4, max_depth=2, random_state=42, n_jobs=1)
    modelo.fit(X, y)
    modelo.save_model(str(pasta / "modelo_saudeIA.json"))
    joblib.dump(tfidf, pasta / "vetorizador_saudeIA.pkl")
    joblib.dump(encoder, pasta / "encoderY_saudeIA.pkl")


def test_arquivos_gerados(tmp_path):
    preparar_modelos_teste(tmp_path)
    assert (tmp_path / "modelo_saudeIA.json").is_file()
    assert (tmp_path / "vetorizador_saudeIA.pkl").is_file()
    assert (tmp_path / "encoderY_saudeIA.pkl").is_file()


def test_carregamento_e_previsao(tmp_path):
    preparar_modelos_teste(tmp_path)
    ia = DiagnosticoIA(tmp_path)
    classe, confianca = ia.prever("vibracao motor")
    assert isinstance(classe, str) and classe
    assert 0 <= confianca <= 1


def test_texto_vazio(tmp_path):
    preparar_modelos_teste(tmp_path)
    ia = DiagnosticoIA(tmp_path)
    with pytest.raises(ValueError, match="não vazio"):
        ia.prever("   ")


def test_arquivos_ausentes(tmp_path):
    with pytest.raises(FileNotFoundError, match="Arquivos do modelo ausentes"):
        DiagnosticoIA(tmp_path)
