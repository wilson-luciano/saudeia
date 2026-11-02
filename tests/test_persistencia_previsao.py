"""Testes rápidos da persistência e da previsão — Aula 03."""

import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from services.persistencia_ml import carregar_componentes, salvar_componentes
from services.previsao_ml import prever_texto


def componentes_exemplo():
    # Pequeno conjunto artificial, independente do CSV real do projeto.
    textos = ["febre tosse", "dor cabeça", "ruido motor", "vibracao eixo"]
    classes = ["saude", "saude", "maquina", "maquina"]

    vetorizador = TfidfVectorizer()
    X = vetorizador.fit_transform(textos)
    encoder = LabelEncoder()
    y = encoder.fit_transform(classes)

    # Modelo mínimo e rápido: serve apenas para testar o fluxo de arquivos.
    modelo = DummyClassifier(strategy="most_frequent")
    modelo.fit(X, y)
    return {"modelo": modelo, "vetorizador": vetorizador, "encoder": encoder}


def test_salvar_e_carregar(tmp_path):
    # Salva em pasta temporária: não altera os modelos do aluno.
    salvar_componentes(componentes_exemplo(), tmp_path)
    for nome in ("modelo.joblib", "vetorizador.joblib", "encoder.joblib"):
        assert (tmp_path / nome).is_file()
    recuperados = carregar_componentes(tmp_path)
    assert set(recuperados) == {"modelo", "vetorizador", "encoder"}


def test_previsao_com_modelo_carregado(tmp_path):
    salvar_componentes(componentes_exemplo(), tmp_path)
    recuperados = carregar_componentes(tmp_path)
    classe = prever_texto("ruido motor", recuperados)
    assert classe in set(recuperados["encoder"].classes_)


def test_texto_vazio():
    # A validação acontece antes de tentar usar os objetos do modelo.
    with pytest.raises(ValueError, match="não vazio"):
        prever_texto("   ", {})


def test_arquivos_ausentes(tmp_path):
    # Deve informar o problema em vez de tentar carregar arquivos inexistentes.
    with pytest.raises(FileNotFoundError, match="Arquivos ausentes"):
        carregar_componentes(tmp_path)
