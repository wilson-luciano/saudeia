"""Aula 05: valida memória e interface sem chamar a API OpenAI."""
from fastapi.testclient import TestClient
import main_api


def test_limite_e_instrucao():
    mensagens = [main_api.instrucao_sistema("Exemplo")]
    mensagens += [{"role": "user", "content": str(i)} for i in range(20)]
    resultado = main_api.limitar_historico(mensagens)
    assert resultado[0]["role"] == "system"
    assert len(resultado) == 11
    assert resultado[-1]["content"] == "19"


def test_dialogo_reutiliza_contexto(monkeypatch):
    chamadas = []

    def simular(self, mensagens):
        chamadas.append([m.copy() for m in mensagens])
        return "Resposta didática simulada"

    monkeypatch.setattr(main_api.AgenteSaudeIA, "perguntar_chat", simular)
    with TestClient(main_api.app) as cliente:
        r1 = cliente.post("/chat_agente", data={"pergunta": "Primeira pergunta", "doenca": "Exemplo"})
        r2 = cliente.post("/chat_agente", data={"pergunta": "Resuma sua resposta", "doenca": "Exemplo"})
        sessao = cliente.cookies.get("session_id")
    assert r1.status_code == r2.status_code == 200
    assert "Histórico da conversa" in r2.text
    assert "Primeira pergunta" in r2.text
    assert "Resuma sua resposta" in r2.text
    assert any(m["role"] == "assistant" for m in chamadas[1])
    main_api.chat_sessions.pop(sessao, None)


def test_contexto_diferente_inicia_nova_conversa(monkeypatch):
    monkeypatch.setattr(main_api.AgenteSaudeIA, "perguntar_chat", lambda self, m: "Simulação")
    with TestClient(main_api.app) as cliente:
        cliente.post("/chat_agente", data={"pergunta": "Pergunta A", "doenca": "Classe A"})
        r = cliente.post("/chat_agente", data={"pergunta": "Pergunta B", "doenca": "Classe B"})
        sessao = cliente.cookies.get("session_id")
    assert "Pergunta A" not in r.text
    assert "Pergunta B" in r.text
    assert "Classe B" in main_api.chat_sessions[sessao][0]["content"]
    main_api.chat_sessions.pop(sessao, None)


def test_nova_conversa_apaga_memoria(monkeypatch):
    """Reiniciar apaga a sessão anterior, cria outra e mantém o chat na página."""
    monkeypatch.setattr(main_api.AgenteSaudeIA, "perguntar_chat", lambda self, m: "Simulação")
    with TestClient(main_api.app) as cliente:
        cliente.post("/chat_agente", data={"pergunta": "Teste", "doenca": "Exemplo"})
        sessao_anterior = cliente.cookies.get("session_id")
        assert sessao_anterior in main_api.chat_sessions
        resposta = cliente.post("/nova_conversa", data={"doenca": "Exemplo"})
        assert resposta.status_code == 200
        assert "Nova conversa iniciada" in resposta.text
        assert "Próxima pergunta ao agente" in resposta.text
        assert "Teste" not in resposta.text
        assert sessao_anterior not in main_api.chat_sessions
        nova_sessao = cliente.cookies.get("session_id")
        assert nova_sessao is not None and nova_sessao != sessao_anterior


def test_nova_conversa_sem_classificacao_exibe_painel(monkeypatch):
    """Mesmo um POST direto sem campo doenca não deve esconder o chat."""
    with TestClient(main_api.app) as cliente:
        resposta = cliente.post("/nova_conversa")
    assert resposta.status_code == 200
    assert "Nova conversa iniciada" in resposta.text
    assert "Próxima pergunta ao agente" in resposta.text


def test_falha_nao_registra_pergunta(monkeypatch):
    def falhar(self, mensagens):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr(main_api.AgenteSaudeIA, "perguntar_chat", falhar)
    with TestClient(main_api.app) as cliente:
        resposta = cliente.post("/chat_agente", data={"pergunta": "Falhou", "doenca": "Exemplo"})
        sessao = cliente.cookies.get("session_id")
    assert resposta.status_code == 200
    assert "Agente indisponível" in resposta.text
    assert all(m["content"] != "Falhou" for m in main_api.chat_sessions[sessao])
    main_api.chat_sessions.pop(sessao, None)


def test_validacao_pergunta_vazia_nao_chama_api(monkeypatch):
    def nao_chamar(self, mensagens):
        raise AssertionError("Não deveria consultar API")

    monkeypatch.setattr(main_api.AgenteSaudeIA, "perguntar_chat", nao_chamar)
    with TestClient(main_api.app) as cliente:
        resposta = cliente.post("/chat_agente", data={"pergunta": "   ", "doenca": "Exemplo"})
        sessao = cliente.cookies.get("session_id")
    assert resposta.status_code == 200
    assert "Digite uma pergunta" in resposta.text
    main_api.chat_sessions.pop(sessao, None)


def test_html_escapa_conteudo_do_modelo(monkeypatch):
    monkeypatch.setattr(main_api.AgenteSaudeIA, "perguntar_chat", lambda self, m: "<script>alert(1)</script>")
    with TestClient(main_api.app) as cliente:
        resposta = cliente.post("/chat_agente", data={"pergunta": "Teste", "doenca": "Exemplo"})
        sessao = cliente.cookies.get("session_id")
    assert "&lt;script&gt;" in resposta.text
    assert "<script>alert(1)</script>" not in resposta.text
    main_api.chat_sessions.pop(sessao, None)


def test_pergunta_seguinte_no_mesmo_painel(monkeypatch):
    """O usuário continua perguntando dentro do histórico, sem retornar ao topo."""
    monkeypatch.setattr(main_api.AgenteSaudeIA, "perguntar_chat", lambda self, m: "Resposta simulada")
    with TestClient(main_api.app) as cliente:
        resposta = cliente.post("/chat_agente", data={"pergunta": "Como funciona?", "doenca": "Exemplo"})
        sessao = cliente.cookies.get("session_id")
    assert resposta.status_code == 200
    assert 'action="/chat_agente"' in resposta.text
    assert 'action="/nova_conversa"' in resposta.text
    assert "Como funciona?" in resposta.text
    assert "Resposta simulada" in resposta.text
    main_api.chat_sessions.pop(sessao, None)


def test_instrucao_permite_explicacoes_gerais():
    """Não confunde educação sobre tratamento com prescrição individual."""
    instrucao = main_api.instrucao_sistema("Exemplo")["content"]
    assert "opções gerais de tratamento" in instrucao
    assert "Não prescreva" in instrucao


def test_prg_post_redireciona_para_get_sem_reenviar(monkeypatch):
    """O POST devolve 303; atualizar GET não repete a chamada ao agente."""
    chamadas = []
    def simular(self, mensagens):
        chamadas.append(mensagens)
        return "Resposta PRG"
    monkeypatch.setattr(main_api.AgenteSaudeIA, "perguntar_chat", simular)
    with TestClient(main_api.app) as cliente:
        resposta = cliente.post("/chat_agente", data={"pergunta": "Uma pergunta", "doenca": "Exemplo"}, follow_redirects=False)
        assert resposta.status_code == 303
        assert resposta.headers["location"] == "/conversa#conversa-agente"
        pagina = cliente.get("/conversa")
        assert pagina.status_code == 200
        assert "Uma pergunta" in pagina.text
        assert "Resposta PRG" in pagina.text
        cliente.get("/conversa")
        assert len(chamadas) == 1
        sessao = cliente.cookies.get("session_id")
    main_api.chat_sessions.pop(sessao, None)
    main_api.chat_views.pop(sessao, None)


def test_prg_nova_conversa_redireciona_e_preserva_aviso():
    with TestClient(main_api.app) as cliente:
        resposta = cliente.post("/nova_conversa", data={"doenca": "Exemplo"}, follow_redirects=False)
        assert resposta.status_code == 303
        pagina = cliente.get("/conversa")
        assert "Nova conversa iniciada" in pagina.text
        assert "Próxima pergunta ao agente" in pagina.text
        sessao = cliente.cookies.get("session_id")
    main_api.chat_views.pop(sessao, None)


def test_get_em_rotas_antigas_nao_retorna_405():
    with TestClient(main_api.app) as cliente:
        for rota in ("/chat_agente", "/nova_conversa"):
            resposta = cliente.get(rota, follow_redirects=False)
            assert resposta.status_code == 303
            assert resposta.headers["location"].startswith("/conversa")
