"""Testes HTTP: dependem dos três arquivos de modelo da Aula 03."""
# TestClient simula um navegador/cliente HTTP sem iniciar o Uvicorn.
from fastapi.testclient import TestClient
# Importar main_api cria a aplicação e carrega o modelo salvo na Aula 03.
import main_api


def test_pagina_inicial():
    # Context manager abre e fecha o cliente de teste corretamente.
    with TestClient(main_api.app) as cliente:
        resposta = cliente.get("/")
    assert resposta.status_code == 200
    assert "SaudeIA" in resposta.text


def test_api_saude():
    """Verifica se a rota de monitoramento retorna JSON e status HTTP 200."""
    with TestClient(main_api.app) as cliente:
        resposta = cliente.get("/api/saude")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "online"


def test_api_prever_simulada(monkeypatch):
    """Testa a API sem depender da previsão real do XGBoost."""
    # monkeypatch troca temporariamente ia.prever por uma função controlada.
    monkeypatch.setattr(main_api.ia, "prever", lambda texto: ("Classe de teste", 0.85))
    with TestClient(main_api.app) as cliente:
        resposta = cliente.post("/api/prever", json={"texto": "febre e tosse"})
    assert resposta.status_code == 200
    assert resposta.json()["classe_prevista"] == "Classe de teste"
    # 0,85 deve ser convertido pelo backend em 85,0%.
    assert resposta.json()["confianca_percentual"] == 85.0


def test_api_prever_vazia():
    """Uma entrada apenas com espaços deve ser rejeitada com HTTP 422."""
    with TestClient(main_api.app) as cliente:
        resposta = cliente.post("/api/prever", json={"texto": "  "})
    assert resposta.status_code == 422


def test_diagnosticar_formulario(monkeypatch):
    """Verifica o POST de formulário HTML, diferente do JSON da API."""
    # A simulação torna o teste independente da classe prevista no treinamento.
    monkeypatch.setattr(main_api.ia, "prever", lambda texto: ("Classe de teste", 0.85))
    with TestClient(main_api.app) as cliente:
        resposta = cliente.post("/diagnosticar", data={"sintomas": "febre e tosse"})
    assert resposta.status_code == 200
    assert "Classe de teste" in resposta.text


def test_documentacao_schema_previsao():
    """Swagger deve declarar os três campos da resposta HTTP 200."""
    with TestClient(main_api.app) as cliente:
        resposta = cliente.get("/openapi.json")
    assert resposta.status_code == 200
    # O OpenAPI descreve os campos que aparecerão no Swagger /docs.
    esquema = resposta.json()["components"]["schemas"]["SaidaPrevisao"]
    assert {"classe_prevista", "confianca_percentual", "aviso"} <= set(esquema["properties"])
