"""SaudeIA: servidor FastAPI, inferência local e conversas com IA.

Este arquivo coordena a interface HTML, o classificador da Aula 03,
o histórico CSV, o QR Code e as rotas de comunicação externa.
"""
# Biblioteca padrão: escreve as classificações em uma tabela CSV.
import csv
# Registra a data e a hora de cada classificação.
from datetime import datetime
# Manipula caminhos de arquivos independentemente da pasta atual do terminal.
from pathlib import Path
# Descobre o endereço IPv4 da máquina para gerar o QR Code.
import socket
# Cria identificadores únicos para sessões de conversa.
import uuid

# Gera a imagem QR Code que aponta para o servidor na rede local.
import qrcode
# FastAPI define rotas; Form recebe formulários; Request representa a requisição.
# HTTPException devolve erros HTTP controlados, como 422.
from fastapi import FastAPI, Form, HTTPException, Request
# HTMLResponse identifica rotas que devolvem páginas HTML.
from fastapi.responses import HTMLResponse, RedirectResponse
# StaticFiles publica CSS, favicon e QR Code em /static.
from fastapi.staticfiles import StaticFiles
# Jinja2Templates renderiza index.html com valores enviados pelo Python.
from fastapi.templating import Jinja2Templates
# Cliente da API externa usado SOMENTE nas rotas de conversa.
from openai import OpenAI
# Pydantic descreve e valida os dados JSON de entrada e saída.
from pydantic import BaseModel, Field

# Reutiliza o classificador treinado e salvo na Aula 03 (sem API externa).
from services.diagnosticoIA import DiagnosticoIA
# Importa o serviço separado que conversa com o modelo de linguagem.
from services.agenteIA import AgenteSaudeIA

# __file__ aponta para este arquivo; .parent identifica a raiz do projeto.
BASE_DIR = Path(__file__).resolve().parent
# Caminhos absolutos evitam erros ao executar a aplicação em outro diretório.
STATIC_DIR = BASE_DIR / "static"
# Cria a pasta static se necessário, sem erro caso já exista.
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = BASE_DIR / "templates"
HISTORICO = BASE_DIR / "historico.csv"

# A aplicação é criada uma única vez, antes dos decoradores de rotas.
app = FastAPI(title="SaudeIA", version="1.0")
# Faz /static/style.css e /static/qrcode.png ficarem acessíveis ao navegador.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# Define onde estão as páginas HTML interpretadas pelo Jinja2.
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# A classificação local usa os três arquivos persistidos na Aula 03.
# Instancia o classificador com os três artefatos já salvos; NÃO treina novamente.
ia = DiagnosticoIA(str(BASE_DIR / "model"))
# Armazenamento em RAM: demonstrativo, sem persistência entre reinícios.
chat_sessions: dict[str, list[dict[str, str]]] = {}
# Estado visual temporário associado à sessão, para a navegação POST/Redirect/GET.
chat_views: dict[str, dict[str, str]] = {}
MAX_MENSAGENS = 10  # Cinco pares pergunta/resposta, quando completos.
MAX_PERGUNTA = 1000



def descobrir_ip_local():
    """Descobre o IP IPv4 da rede; fallback para localhost se não houver rede."""
    try:
        # Um socket UDP permite identificar a interface de rede escolhida pelo SO.
        # connect() aqui não envia sintomas nem faz uma chamada à API da OpenAI.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as conexao:
            conexao.connect(("8.8.8.8", 80))
            # Obtém o endereço local da interface de rede selecionada.
            return conexao.getsockname()[0]
    except OSError:
        # Fallback: acessível no computador, mas não necessariamente no celular.
        return "127.0.0.1"


def criar_qrcode():
    # O QR Code deve usar o IPv4 real, nunca 0.0.0.0.
    endereco = f"http://{descobrir_ip_local()}:8000"
    # O navegador poderá ler a imagem pela rota /static/qrcode.png.
    qrcode.make(endereco).save(STATIC_DIR / "qrcode.png")
    print("QR Code criado para:", endereco)


criar_qrcode()


def get_session_id(request: Request) -> str:
    """Recupera o identificador do navegador ou cria um UUID novo."""
    identificador = request.cookies.get("session_id", "")
    try:
        # Só aceita UUIDs válidos, evitando identificadores arbitrários.
        return str(uuid.UUID(identificador))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid4())


def instrucao_sistema(doenca: str) -> dict[str, str]:
    """Define limites e contexto educacional da conversa."""
    classificacao = doenca.strip()[:120] or "não informada"
    return {
        "role": "system",
        "content": (
            "Você é um assistente educacional sobre saúde. "
            "Explique didaticamente sintomas, exames, prevenção e opções gerais de tratamento. "
            "Ao perguntarem sobre tratamentos, descreva as abordagens habituais e seus objetivos. "
            "A classificação é experimental e NÃO é diagnóstico: não pressuponha que "
            "o usuário tenha a condição classificada. "
            "Não prescreva medicamentos, doses nem condutas individualizadas. "
            "Não solicite dados pessoais. Em situações urgentes, recomende atendimento médico. "
            f"Classificação experimental atual: {classificacao}."
        ),
    }


def limitar_historico(mensagens: list[dict[str, str]]) -> list[dict[str, str]]:
    """Preserva a instrução system e até dez mensagens recentes."""
    return [mensagens[0]] + mensagens[1:][-MAX_MENSAGENS:]


def mensagens_visiveis(mensagens: list[dict[str, str]]) -> list[dict[str, str]]:
    """Não apresenta a instrução interna system na interface."""
    return [m for m in mensagens if m["role"] in ("user", "assistant")]


def contexto_pagina(request: Request, **dados):
    """Renderiza a página mantendo os valores recebidos do backend."""
    return templates.TemplateResponse(request=request, name="index.html", context=dados)


# GET /: exibe a página inicial sem enviar dados para classificação.
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Sintaxe atual do Starlette: request e name são argumentos nomeados.
    return templates.TemplateResponse(request=request, name="index.html")


# Compatibilidade de navegação: GET não deve executar operações de POST.
# O redirecionamento evita 405 em links ou histórico do navegador,
# mas não implementa PRG para os formulários antigos.
@app.get("/diagnosticar", include_in_schema=False)
@app.get("/chat", include_in_schema=False)
async def redirecionar_formularios_antigos():
    """Redireciona acessos GET às rotas de processamento para a página inicial."""
    return RedirectResponse(url="/", status_code=303)


# POST /diagnosticar: recebe o campo "sintomas" do formulário HTML.
@app.post("/diagnosticar", response_class=HTMLResponse)
async def diagnosticar(request: Request, sintomas: str = Form("")):
    # strip() retira espaços nas extremidades; split() separa as palavras.
    # O exemplo exige duas palavras para evitar entradas vazias ou muito curtas.
    if len(sintomas.strip().split()) < 2:
        # Devolve a mesma página, mas com uma mensagem de validação.
        return templates.TemplateResponse(
            request=request, name="index.html",
            context={"erro": "Digite pelo menos dois sintomas!", "sintomas": sintomas},
        )

    # prever() retorna (classe prevista, probabilidade entre 0 e 1).
    # A inferência acontece localmente, sem consumir créditos da OpenAI.
    doenca, prob = ia.prever(sintomas)
    # CSV histórico local: somente dados fictícios durante as aulas.
    # Se o CSV ainda não existe ou está vazio, será necessário criar o cabeçalho.
    arquivo_novo = not HISTORICO.exists() or HISTORICO.stat().st_size == 0
    # Modo "a" acrescenta uma linha sem apagar as classificações anteriores.
    with HISTORICO.open("a", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        # Cabeçalho apenas na primeira gravação.
        if arquivo_novo:
            writer.writerow(["data", "sintomas", "doenca", "prob"])
        # Uma linha por classificação; use somente textos fictícios em aula.
        writer.writerow([datetime.now().isoformat(timespec="seconds"), sintomas, doenca, prob])

    # context leva os valores para {{ doenca }}, {{ prob }} e {{ sintomas }}.
    # Multiplicamos por 100 para mostrar a probabilidade em porcentagem.
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={"doenca": doenca, "prob": round(float(prob) * 100, 2), "sintomas": sintomas},
    )


# Assistente simples: cada pergunta é enviada sem histórico de mensagens anteriores.
@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, pergunta: str = Form(""), doenca: str = Form("")):
    if not pergunta.strip() or len(pergunta.strip()) > MAX_PERGUNTA:
        return contexto_pagina(request, doenca=doenca, erro="Informe uma pergunta de até 1000 caracteres.")
    # Prompt: instruções que contextualizam a pergunta enviada ao LLM.
    prompt = (
        "Você é um assistente educacional sobre saúde. "
        "Explique opções gerais de tratamento, prevenção e exames quando perguntado, "
        "sem prescrever doses ou condutas individualizadas. "
        "A classificação é experimental, não confirma diagnóstico. "
        "Oriente a procurar profissionais habilitados para decisões clínicas. "
        f"Classificação do sistema: {doenca}. Pergunta: {pergunta}"
    )
    try:
        # Criação tardia: sem chave, a classificação local continua disponível.
        cliente = OpenAI()
        # Esta linha faz uma chamada de rede que pode consumir créditos da API.
        resposta = cliente.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )
        # Extrai o conteúdo textual da primeira alternativa devolvida.
        texto = resposta.choices[0].message.content or "Sem resposta textual."
    except Exception as erro:
        # Não derruba o classificador local caso a API externa falhe.
        print("Falha no assistente:", type(erro).__name__, str(erro))
        texto = "⚠️ Assistente indisponível. Verifique a chave, a conexão e o faturamento da API."
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={"doenca": doenca, "prob": "", "resposta": texto},
    )


# A rota GET é a única responsável por apresentar o histórico após um POST.
@app.get("/conversa", response_class=HTMLResponse)
async def ver_conversa(request: Request):
    """Mostra a conversa existente; atualizar esta URL nunca reenvia o POST."""
    session_id = get_session_id(request)
    estado = chat_views.get(session_id, {})
    historico = chat_sessions.get(session_id, [])
    pagina = contexto_pagina(
        request,
        doenca=estado.get("doenca", ""),
        historico_agente=mensagens_visiveis(historico),
        erro_agente=estado.get("erro_agente"),
        aviso_conversa=estado.get("aviso_conversa"),
    )
    pagina.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return pagina


# Compatibilidade com links antigos: GET não altera a memória.
@app.get("/chat_agente")
@app.get("/nova_conversa")
async def redirecionar_links_antigos():
    return RedirectResponse(url="/conversa#conversa-agente", status_code=303)


# POST processa e armazena; depois redireciona para GET /conversa.
@app.post("/chat_agente")
async def chat_agente(
    request: Request,
    pergunta: str = Form(""),
    doenca: str = Form(""),
):
    session_id = get_session_id(request)
    pergunta = pergunta.strip()
    doenca = doenca.strip()[:120]

    if session_id not in chat_sessions:
        chat_sessions[session_id] = [instrucao_sistema(doenca)]
    historico = chat_sessions[session_id]
    nova_instrucao = instrucao_sistema(doenca)
    if historico[0] != nova_instrucao:
        historico = [nova_instrucao]
        chat_sessions[session_id] = historico

    erro_agente = ""
    if not pergunta or len(pergunta) > MAX_PERGUNTA:
        erro_agente = f"Digite uma pergunta com 1 a {MAX_PERGUNTA} caracteres."
    else:
        tentativa = limitar_historico(historico + [{"role": "user", "content": pergunta}])
        try:
            resposta = AgenteSaudeIA().perguntar_chat(tentativa)
            if not isinstance(resposta, str) or not resposta.strip():
                raise ValueError("Resposta textual vazia")
            chat_sessions[session_id] = limitar_historico(
                tentativa + [{"role": "assistant", "content": resposta}]
            )
        except Exception as erro:
            print("Falha no agente:", type(erro).__name__)
            erro_agente = "Agente indisponível. Verifique conexão, chave e créditos da API."

    chat_views[session_id] = {"doenca": doenca, "erro_agente": erro_agente}
    pagina = RedirectResponse(url="/conversa#conversa-agente", status_code=303)
    pagina.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return pagina


@app.post("/nova_conversa")
async def nova_conversa(request: Request, doenca: str = Form("")):
    """Apaga o histórico anterior e inicia nova sessão sem sair do chat."""
    session_id = get_session_id(request)
    chat_sessions.pop(session_id, None)
    chat_views.pop(session_id, None)
    nova_sessao = str(uuid.uuid4())
    chat_views[nova_sessao] = {
        "doenca": doenca.strip()[:120],
        "aviso_conversa": "Nova conversa iniciada. Digite sua pergunta abaixo.",
    }
    pagina = RedirectResponse(url="/conversa#conversa-agente", status_code=303)
    pagina.set_cookie("session_id", nova_sessao, httponly=True, samesite="lax")
    return pagina


# Endpoint de saúde do SERVIDOR (não é avaliação clínica de saúde).
@app.get("/api/saude")
def api_saude():
    return {"status": "online", "servico": "SaudeIA"}


# Pydantic rejeita automaticamente JSON com texto ausente ou muito longo.
class EntradaPrevisao(BaseModel):
    # A validação de duas palavras também é feita na função api_prever().
    texto: str = Field(min_length=1, max_length=1000)


class SaidaPrevisao(BaseModel):
    """Esquema JSON mostrado no Swagger e devolvido ao cliente."""
    classe_prevista: str           # Rótulo devolvido pelo classificador.
    confianca_percentual: float    # Probabilidade estimada × 100 (não é acurácia).
    aviso: str                     # Alerta sobre a finalidade educacional.


# POST /api/prever: recebe JSON, não um formulário HTML.
# response_model determina o formato validado e documentado no Swagger.
@app.post("/api/prever", response_model=SaidaPrevisao)
def api_prever(entrada: EntradaPrevisao):
    # Remove espaços antes de validar e enviar o texto ao classificador.
    texto = entrada.texto.strip()
    if len(texto.split()) < 2:
        # Código HTTP 422 informa que o corpo JSON não passou na validação.
        raise HTTPException(status_code=422, detail="Informe pelo menos duas palavras.")
    # Mesmo classificador utilizado pelo formulário /diagnosticar.
    classe, prob = ia.prever(texto)
    return SaidaPrevisao(
        classe_prevista=classe,
        # Converte de escala 0–1 para percentual, com duas casas decimais.
        confianca_percentual=round(float(prob) * 100, 2),
        aviso="Classificação exclusivamente didática; não constitui diagnóstico médico.",
    )


# Permite iniciar com "python -m main_api"; o comando uvicorn é preferível na aula.
if __name__ == "__main__":
    import uvicorn
    # Execução local. Para acesso via QR Code, use o comando da Seção 9.
    uvicorn.run(app, host="127.0.0.1", port=8000)
