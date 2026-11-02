"""Testes de integração HTTP da aplicação SaudeIA — Aula 04."""

import pytest
from fastapi.testclient import TestClient
import main_api


@pytest.fixture
def cliente(monkeypatch):
    """Cria um cliente HTTP sem carregar os arquivos .joblib reais."""
    # Injeta componentes de teste na dependência da rota.
    main_api.app.dependency_overrides[main_api.obter_componentes] = (
        lambda: {"simulado": True}
    )

    # Simula a função de previsão; não precisa treinar XGBoost.
    monkeypatch.setattr(
        main_api,
        "prever_texto",
        lambda texto, componentes: "classe_teste",
    )

    with TestClient(main_api.app) as client:
        yield client

    # Remove a substituição para não afetar outros testes.
    main_api.app.dependency_overrides.clear()


def test_pagina_inicial(cliente):
    resposta = cliente.get("/")
    assert resposta.status_code == 200
    assert "SaudeIA" in resposta.text


def test_servidor_online(cliente):
    resposta = cliente.get("/api/saude")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "online"


def test_previsao_valida(cliente):
    resposta = cliente.post("/api/prever", json={"texto": "febre"})
    assert resposta.status_code == 200
    assert resposta.json()["classe_prevista"] == "classe_teste"
    assert "diagnóstico médico" in resposta.json()["aviso"]


def test_texto_vazio(cliente):
    resposta = cliente.post("/api/prever", json={"texto": "   "})
    assert resposta.status_code == 422


def test_texto_muito_longo(cliente):
    resposta = cliente.post("/api/prever", json={"texto": "a" * 1001})
    assert resposta.status_code == 422
