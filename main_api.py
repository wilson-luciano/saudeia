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
from fastapi.responses import HTMLResponse
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
chat_sessions = {}  # Memória temporária: perde-se ao reiniciar o servidor.


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


def get_session_id(request: Request):
    # Reaproveita o cookie existente; sem ele, gera uma nova sessão.
    return request.cookies.get("session_id") or str(uuid.uuid4())


# GET /: exibe a página inicial sem enviar dados para classificação.
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Sintaxe atual do Starlette: request e name são argumentos nomeados.
    return templates.TemplateResponse(request=request, name="index.html")


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
async def chat(request: Request, pergunta: str = Form(...), doenca: str = Form(...)):
    # Prompt: instruções que contextualizam a pergunta enviada ao LLM.
    prompt = (
        "Você é um assistente educacional sobre saúde. "
        "Não confirme diagnósticos nem prescreva tratamentos. "
        "Oriente a procurar profissionais habilitados. "
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


# Agente: envia mensagens anteriores para manter o contexto da conversa.
@app.post("/chat_agente", response_class=HTMLResponse)
async def chat_agente(request: Request, pergunta: str = Form(...), doenca: str = Form(...)):
    # Um cookie identifica o histórico da conversa deste navegador.
    session_id = get_session_id(request)
    # Cria uma instrução "system" quando a sessão ainda não existe.
    if session_id not in chat_sessions:
        chat_sessions[session_id] = [{
            "role": "system",
            "content": (
                "Você é um agente conversacional educacional sobre saúde. "
                f"Contexto: {doenca}. Não confirme diagnósticos nem prescreva tratamentos. "
                "Oriente a procurar avaliação profissional."
            ),
        }]
    historico = chat_sessions[session_id]
    # Acrescenta a pergunta atual à sequência de mensagens.
    historico.append({"role": "user", "content": pergunta})
    # Mantém instrução system e até dez mensagens recentes.
    chat_sessions[session_id] = [historico[0]] + historico[1:][-10:]
    try:
        # O serviço separado chama a API externa e devolve a resposta textual.
        agente = AgenteSaudeIA()
        resposta = agente.perguntar_chat(chat_sessions[session_id])
        # Salva a resposta para que a próxima pergunta tenha contexto.
        chat_sessions[session_id].append({"role": "assistant", "content": resposta})
    except Exception as erro:
        # Falha da API não deve impedir o uso do classificador local.
        print("Falha no agente:", type(erro).__name__, str(erro))
        resposta = "⚠️ Agente indisponível. Verifique a chave, a conexão e o faturamento da API."
    # Renderiza a página com a pergunta e a resposta da sessão atual.
    pagina = templates.TemplateResponse(
        request=request, name="index.html",
        context={"doenca": doenca, "ultima_pergunta": pergunta, "ultima_resposta": resposta},
    )
    # Cookie HTTP mantém a sessão entre requisições; não guarda a chave da API.
    pagina.set_cookie("session_id", session_id, httponly=True, samesite="lax")
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
