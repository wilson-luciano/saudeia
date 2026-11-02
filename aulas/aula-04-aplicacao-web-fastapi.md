# Aula 04 — Aplicação Web do SaudeIA com FastAPI, API e QR Code

**Disciplina:** Programação para Engenharia Mecânica — 2026.2  
**Projeto Avaliativo 3:** SaudeIA — Inteligência Artificial Aplicada à Engenharia  
**Ambiente:** Windows · VSCode · PowerShell · Python 3.12 · `.venv` · FastAPI · HTML · CSS

> **Finalidade exclusivamente didática:** este sistema classifica textos; não realiza diagnóstico médico, não recomenda tratamento e não substitui profissionais de saúde. Utilize somente sintomas fictícios nas atividades.

## 1. Objetivos da aula

Ao concluir, você será capaz de: (1) explicar frontend, backend, API e agente conversacional; (2) construir o servidor FastAPI com o `main_api.py`; (3) conectar a interface HTML ao classificador local treinado nas Aulas 02 e 03; (4) configurar com segurança uma chave de API externa; (5) executar a aplicação no computador e no celular via QR Code; (6) validar as rotas e publicar o código sem segredos no GitHub.

**Pré-requisito:** os arquivos da Aula 03 devem existir em `model/`: `modelo_saudeIA.json`, `vetorizador_saudeIA.pkl` e `encoderY_saudeIA.pkl`. Não treinaremos novamente o modelo nesta aula.

## 2. Arquitetura da aplicação

- **Frontend:** `templates/index.html` e `static/style.css` apresentam formulários e respostas.
- **Backend:** `main_api.py` recebe requisições HTTP com FastAPI e devolve páginas HTML ou JSON.
- **Classificador local:** `services/diagnosticoIA.py` carrega XGBoost, TF-IDF e LabelEncoder. **Não consome API da OpenAI.**
- **Assistente simples:** `/chat` envia uma pergunta ao modelo de linguagem, sem memória persistente da conversa.
- **Agente conversacional:** `/chat_agente` usa `services/agenteIA.py` e recebe mensagens anteriores da sessão. Nesta implementação, é um agente conversacional simples, **sem ferramentas autônomas**.
- **API JSON:** `/api/saude` verifica o servidor e `/api/prever` classifica textos enviados por outros programas.

<p align="center"><img src="../assets/fluxogramas/arquitetura-web-saudeia.svg" alt="Arquitetura web do SaudeIA" width="760"></p>
<p align="center"><em>Figura 1 — Comunicação entre interface, servidor e componentes de classificação.</em></p>

**GET** solicita um recurso, como a página inicial. **POST** envia dados para processamento, como o formulário de sintomas. **JSON** é um formato de dados usado por APIs. A FastAPI disponibiliza documentação interativa em `/docs`.

## 3. Preparação e dependências

No PowerShell integrado ao VSCode, execute:

```powershell
cd C:\Projetos\SaudeIA
.\.venv\Scripts\Activate.ps1

# Confirma os três arquivos produzidos na Aula 03.
Test-Path .\model\modelo_saudeIA.json
Test-Path .\model\vetorizador_saudeIA.pkl
Test-Path .\model\encoderY_saudeIA.pkl

# Confirma a interface existente.
Test-Path .\templates\index.html
Test-Path .\static\style.css
```

Os cinco testes devem retornar `True`. Se os arquivos de modelo estiverem ausentes, conclua a persistência da Aula 03 antes de prosseguir.

Instale as bibliotecas necessárias no `.venv`:

```powershell
python -m pip install fastapi "uvicorn[standard]" jinja2 python-multipart qrcode[pil] openai httpx pytest
python -m pip show openai
```

O `python-multipart` permite ler os formulários HTML. `qrcode[pil]` gera o QR Code. `openai` conecta os recursos conversacionais à API externa. `httpx` e `pytest` permitem testar as rotas. Os pacotes de Machine Learning já foram instalados nas aulas anteriores.

## 4. Criar o módulo do agente conversacional

**Agente conversacional** é o componente que envia ao modelo de linguagem uma sequência de mensagens (`system`, `user`, `assistant`). O `main_api.py` administra o histórico temporário e o `agenteIA.py` realiza a chamada externa. Assim, a classificação XGBoost e a conversa com LLM permanecem separadas.

Crie o arquivo, caso ainda não exista:

```powershell
New-Item -ItemType Directory -Force .\services | Out-Null
if (-not (Test-Path .\services\agenteIA.py)) {
    New-Item -ItemType File .\services\agenteIA.py | Out-Null
}
code .\services\agenteIA.py
```

**Arquivo:** `services/agenteIA.py`. Antes de copiar, observe que `__init__()` prepara o cliente de comunicação, enquanto `perguntar_chat()` recebe uma lista de mensagens e devolve a resposta. Copie e salve:

```python
"""Cliente de linguagem para o agente conversacional educacional."""
# Cliente oficial: realiza requisições HTTPS à API externa.
from openai import OpenAI


class AgenteSaudeIA:
    """Encaminha mensagens ao modelo de linguagem pela API externa."""

    # O construtor é chamado quando o backend precisa de uma conversa.
    def __init__(self):
        # OPENAI_API_KEY é lida automaticamente do ambiente.
        self.client = OpenAI()

    # Método responsável por enviar a conversa e receber a resposta.
    def perguntar_chat(self, mensagens):
        """Recebe lista de mensagens (system/user/assistant) e devolve texto."""
        # model escolhe o LLM; messages preserva a ordem da conversa.
        # Esta chamada pode gerar custos na conta de API configurada.
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=mensagens
        )
        # choices[0] seleciona a primeira resposta; content contém o texto.
        return completion.choices[0].message.content or "Sem resposta textual."
```

**Entendendo o código:** os comentários no arquivo explicam a criação do cliente, o envio das mensagens e a leitura da resposta. `from openai import OpenAI` importa o cliente oficial. `class AgenteSaudeIA` organiza a funcionalidade em uma classe; `self.client` guarda o cliente para reutilização. O parâmetro `mensagens` representa a conversa com os papéis `system` (instruções), `user` (pergunta) e `assistant` (resposta anterior). O método `create()` envia a requisição; `choices[0].message.content` extrai o texto. Uma chamada real pode consumir créditos da API.

Teste **sem consumir créditos**, apenas importando a classe:

```powershell
python -c "from services.agenteIA import AgenteSaudeIA; print('Importação OK')"
```

## 5. Chave de API da OpenAI: conceito, instalação e segurança

A **biblioteca Python** (`openai`) não é a **chave de API** (`OPENAI_API_KEY`). A biblioteca permite a comunicação; a chave identifica o projeto autorizado; as chamadas podem gerar custos de uso. Uma assinatura do ChatGPT não equivale a créditos de API. Consulte o painel de consumo e faturamento da conta correspondente.

**Obter a chave:** entre em https://platform.openai.com/api-keys, escolha o projeto e crie uma chave secreta, conforme as opções disponíveis na conta. Não compartilhe a chave. Consulte o uso em https://platform.openai.com/usage.

### 5.1. Verificar a instalação sem revelar o segredo

```powershell
python -m pip show openai

if ($env:OPENAI_API_KEY) {
    Write-Host "Chave configurada nesta sessão."
} else {
    Write-Host "Chave não encontrada nesta sessão."
}

[bool][Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")
```

**Não utilize `echo $env:OPENAI_API_KEY`: esse comando mostra a chave completa na tela.**

### 5.2. Três formas de fornecer a chave

**A — Diretamente no código (não recomendado):** `OpenAI(api_key="SUA_CHAVE")`. É tecnicamente possível, mas expõe o segredo ao compartilhar o arquivo ou publicar no GitHub. **Não adote este método nas atividades.**

**B — Variável de ambiente temporária no PowerShell:**

```powershell
$env:OPENAI_API_KEY = "COLE_SUA_CHAVE_AQUI"
```

Vale apenas para o terminal atual e os processos iniciados a partir dele.

**C — Variável de ambiente persistente do usuário Windows:**

```powershell
[Environment]::SetEnvironmentVariable(
    "OPENAI_API_KEY", "COLE_SUA_CHAVE_AQUI", "User"
)
```

Feche e abra um **novo terminal** antes de testar. Também é possível utilizar `setx OPENAI_API_KEY "COLE_SUA_CHAVE_AQUI"`, que só atualiza os próximos terminais.

**Alternativa para projetos em equipe — arquivo `.env`:** instale `python-dotenv`, crie um `.env` local com `OPENAI_API_KEY=...`, adicione `.env` ao `.gitignore` e carregue com `from dotenv import load_dotenv; load_dotenv()` **antes de instanciar** `OpenAI()`. Disponibilize apenas um `.env.example` sem chave. Neste roteiro, utilizaremos a variável de ambiente do Windows, dispensando dependência adicional.

**Erro `401`:** verifique chave/projeto. **Erro `429` com `insufficient_quota`:** verifique limites, créditos e faturamento. **Erro de conexão:** verifique internet/proxy. Evite colocar dados reais de pacientes nas requisições externas.

## 6. Construir o backend completo

O único arquivo principal do projeto é `main_api.py`. Ele carrega os três componentes persistidos na Aula 03, atende os formulários originais, registra histórico CSV e expõe as rotas JSON. Todas as chamadas de `TemplateResponse()` usam a sintaxe atual: `request=request, name="index.html", context={...}`.

Antes de substituir o arquivo, crie uma cópia de segurança:

```powershell
Copy-Item .\main_api.py .\main_api_backup.py
code .\main_api.py
```

**Como ler o código:** acompanhe os comentários internos, que explicam o propósito de cada biblioteca, a configuração das pastas, o fluxo das rotas e a conversão da probabilidade em percentual. Não é necessário decorar cada instrução: relacione os comentários às etapas do processamento. Os `imports` disponibilizam bibliotecas; `app = FastAPI()` cria o servidor; `app.mount()` publica arquivos estáticos; `Jinja2Templates()` conecta o HTML; e os decoradores `@app.get()` / `@app.post()` definem as rotas. `Form()` recebe campos do formulário, `BaseModel` valida JSON e `TemplateResponse()` envia dados do Python ao HTML.

**Arquivo:** `main_api.py`. Copie o código completo abaixo e salve com `Ctrl+S`:

```python
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
```

**Roteiro de leitura do `main_api.py`:** comece pelas importações e pelos caminhos (`BASE_DIR`); identifique a criação de `app`, o carregamento de `DiagnosticoIA` e a geração do QR Code. Em seguida, acompanhe `GET /`, `POST /diagnosticar`, `POST /chat`, o fluxo PRG das rotas `/chat_agente`, `/nova_conversa` e `/conversa` e, por fim, as rotas JSON. Os comentários dentro do arquivo acompanham esse percurso.

**O que acontece ao clicar em Diagnosticar?** O navegador envia um `POST` contendo o texto dos sintomas. A função da rota valida a entrada, chama `ia.prever()`, grava o histórico em CSV e devolve a página com os valores no `context`. O arquivo de modelo é carregado para inferência; não é treinado novamente.

**Por que usar `request=request`?** É a referência à requisição HTTP exigida pela renderização de templates. `name="index.html"` indica qual página abrir e `context={...}` define os valores que o Jinja2 substituirá no HTML.

### 6.1. Principais rotas

| Rota | Método | Função |
| --- | --- | --- |
| `/` | GET | Exibe a interface principal |
| `/diagnosticar` | POST | Classifica sintomas fictícios e registra o histórico CSV |
| `/diagnosticar` | GET | Redireciona para `/` (HTTP 303) |
| `/chat` | POST | Envia uma pergunta ao assistente simples, sem memória |
| `/chat` | GET | Redireciona para `/` (HTTP 303) |
| `/conversa` | GET | Exibe a conversa associada à sessão atual |
| `/chat_agente` | POST | Processa a pergunta e redireciona para `/conversa` (HTTP 303) |
| `/nova_conversa` | POST | Limpa a conversa da sessão e redireciona (HTTP 303) |
| `/chat_agente` e `/nova_conversa` | GET | Redirecionam para `/conversa` (HTTP 303) |
| `/api/saude` | GET | Verifica se o servidor responde |
| `/api/prever` | POST | Recebe JSON e devolve a classificação |
| `/docs` | GET | Apresenta a documentação automática da API |

### 6.2. Padrão POST/Redirect/GET (PRG)

No agente, o navegador envia `POST /chat_agente`; o servidor processa a pergunta, atualiza o histórico da sessão e responde com **HTTP 303**, apontando para `GET /conversa`. A página de conversa é carregada por uma requisição GET. Assim, atualizar a página não reenvia o formulário anterior.

O botão **Nova conversa** segue o mesmo padrão: `POST /nova_conversa` limpa o histórico da sessão e redireciona para `GET /conversa`. O histórico fica em memória no processo Python; reiniciar o servidor o apaga. A sessão é identificada por um cookie com identificador aleatório, não por dados pessoais.

**Atenção:** `GET /diagnosticar` e `GET /chat` são atalhos de navegação para a página inicial; não executam classificação nem chamadas à IA. O método HTTP importa: `HEAD` não equivale a `GET`.

**Por que o `TemplateResponse()` mudou?** Nas versões atuais do Starlette, a forma recomendada é nomear explicitamente `request` e `name`. O padrão antigo `TemplateResponse("index.html", {"request": request})` pode produzir erro HTTP 500 (`TypeError: unhashable type: 'dict'`).

## 7. Interface HTML e CSS — códigos completos e comentados

A interface utiliza **`templates/index.html`** (estrutura, formulários e variáveis Jinja2) e **`static/style.css`** (cores, espaçamentos e responsividade). Os dois arquivos são apresentados **integralmente**, para que o aluno possa copiar, colar e executar sem procurar trechos em aulas anteriores.

**Integração dos formulários:** a classificação envia `sintomas` para `/diagnosticar`; o assistente simples envia `pergunta` e `doenca` para `/chat`; o agente utiliza `/chat_agente`, com visualização em `/conversa` e botão **Nova conversa** em `/nova_conversa`. Os nomes dos campos HTML devem coincidir com os parâmetros `Form()` do backend. A interface inclui o aviso educacional e o QR Code.

### 7.1. Preparar e abrir os arquivos

No PowerShell integrado ao VSCode:

```powershell
# Execute a partir da raiz do SaudeIA.
cd C:\Projetos\SaudeIA

# Faça cópias de segurança antes de substituir arquivos existentes.
Copy-Item .\templates\index.html .\templates\index_antes_revisao04.html
Copy-Item .\static\style.css .\static\style_antes_revisao04.css

# Abra os arquivos nos quais os códigos serão colados.
code .\templates\index.html
code .\static\style.css
```

As cópias de segurança são locais e não devem ser incluídas no commit. Depois de validar o funcionamento, podem ser removidas. Para que não sejam adicionadas por engano ao Git, confira o `git status` antes do commit.

### 7.2. HTML completo — `templates/index.html`

**Substitua todo o conteúdo do arquivo pelo código abaixo.** Os comentários `<!-- ... -->` identificam os formulários, as variáveis Jinja2 e os elementos da página.

```html
<!doctype html>
<html lang="pt-BR">
<head>
    <!-- Define o conjunto de caracteres, incluindo acentos. -->
    <meta charset="utf-8">
    <!-- Ajusta a largura da página para celulares e tablets. -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SaudeIA</title>

    <!-- CSS e favicon são servidos pela pasta static/ do FastAPI. -->
    <link rel="stylesheet" href="/static/style.css">
    <link rel="icon" href="/static/favicon.ico">
</head>
<body>
    <!-- Agrupa todo o conteúdo visual da aplicação. -->
    <main class="container">
        <h2>🩺 SaudeIA</h2>

        <!-- Jinja2: exibe o erro somente quando o backend envia 'erro'. -->
        {% if erro %}
        <div class="resultado erro" role="alert">{{ erro }}</div>
        {% endif %}

        <!-- FORMULÁRIO DE CLASSIFICAÇÃO: POST /diagnosticar. -->
        <form action="/diagnosticar" method="post">
            <!-- O atributo name identifica o campo recebido por Form(). -->
            <label for="sintomas">Descrição fictícia dos sintomas</label>
            <textarea id="sintomas" name="sintomas"
                      placeholder="Digite os sintomas fictícios...">{{ sintomas or '' }}</textarea>
            <button type="submit">Diagnosticar</button>
        </form>

        <!-- Link GET / limpa o formulário; evita um botão dentro de <a>. -->
        <a class="botao botao-secundario" href="/">Limpar</a>

        <!-- RESULTADO LOCAL: valores fornecidos por ia.prever(). -->
        {% if doenca %}
        <section class="resultado" aria-label="Resultado da classificação">
            <h3>🩺 Classificação experimental</h3>
            <h2>{{ doenca }}</h2>
            <!-- prob pode ser string vazia quando uma rota de chat renderiza a página. -->
            {% if prob is defined and prob != '' and prob is not none %}
            <p><strong>Confiança do classificador:</strong> {{ prob }} %</p>
            {% endif %}
            <p class="aviso-curto">Resultado didático; não constitui diagnóstico médico.</p>
        </section>
        {% endif %}

        <!-- Assistente simples: disponível após a classificação experimental. -->
        {% if doenca %}
        <hr>
        <!-- ASSISTENTE SIMPLES: cada pergunta é independente. -->
        <h3>💬 Converse com o Assistente</h3>
        <form action="/chat" method="post">
            <!-- Envia a classificação atual junto da pergunta. -->
            <input type="hidden" name="doenca" value="{{ doenca }}">
            <label for="pergunta-simples">Pergunta ao assistente</label>
            <textarea id="pergunta-simples" name="pergunta" maxlength="1000"
                      placeholder="Digite sua dúvida..." required></textarea>
            <button type="submit">Perguntar (Simples)</button>
        </form>
        <!-- A rota /chat envia apenas a resposta atual como 'resposta'. -->
        {% if resposta %}
        <div class="resultado" aria-live="polite">{{ resposta }}</div>
        {% endif %}

        {% endif %}

        <!-- O chat continua visível após Nova conversa, mesmo sem classificação. -->
        {% if doenca or aviso_conversa or historico_agente is defined %}
        <hr>
        <!-- CHAT CONTÍNUO: o histórico e a próxima pergunta ficam juntos. -->
        <section class="secao-agente" id="conversa-agente" aria-label="Agente de IA">
            <h3>🤖 Agente de IA — Conversa contínua</h3>
            <p class="aviso-curto">Respostas educativas; classificação experimental não é diagnóstico.</p>
            {% if aviso_conversa %}
            <div class="resultado aviso-conversa" role="status">{{ aviso_conversa }}</div>
            {% endif %}
            {% if erro_agente %}
            <div class="resultado erro" role="alert">{{ erro_agente }}</div>
            {% endif %}
            <div class="chat-box" id="historico-agente" aria-label="Histórico da conversa" aria-live="polite">
                <h3>Histórico da conversa</h3>
                {% if historico_agente %}
                    {% for mensagem in historico_agente %}
                        {% if mensagem.role == 'user' %}
                        <div class="chat-user"><strong>Você:</strong><p>{{ mensagem.content }}</p></div>
                        {% else %}
                        <div class="chat-agent"><strong>Agente:</strong><p>{{ mensagem.content }}</p></div>
                        {% endif %}
                    {% endfor %}
                {% else %}
                    <p class="chat-vazio">Faça a primeira pergunta para iniciar o diálogo.</p>
                {% endif %}
            </div>
            <!-- Formulário permanece visível abaixo das mensagens. -->
            <form action="/chat_agente" method="post" class="form-chat">
                <input type="hidden" name="doenca" value="{{ doenca }}">
                <label for="pergunta-agente">Próxima pergunta ao agente</label>
                <textarea id="pergunta-agente" name="pergunta" maxlength="1000" rows="3"
                          placeholder="Digite sua próxima pergunta..." required></textarea>
                <button type="submit">Enviar pergunta</button>
            </form>
            <!-- Nova conversa não volta à página inicial. -->
            <form action="/nova_conversa" method="post" class="form-nova-conversa">
                <input type="hidden" name="doenca" value="{{ doenca }}">
                <button type="submit" class="botao-nova-conversa">Nova conversa</button>
            </form>
        </section>
        <script>
            // Após renderizar, rola o histórico para a última mensagem.
            const historico = document.getElementById("historico-agente");
            if (historico) historico.scrollTop = historico.scrollHeight;
        </script>
        {% endif %}

        <!-- AVISO OBRIGATÓRIO SOBRE O CARÁTER EDUCACIONAL. -->
        <hr>
        <p class="aviso-legal">
            ⚠️ Este sistema é educacional e não substitui avaliação médica!
            Utilize somente dados fictícios.
        </p>

        <!-- QR Code gerado pelo backend para acesso na mesma rede local. -->
        <div class="qr-code">
            <img src="/static/qrcode.png" width="160"
                 alt="QR Code para acessar o SaudeIA pela rede local">
            <p>Aponte a câmera do celular</p>
        </div>
    </main>
</body>
</html>
```

**Entendendo o HTML:**

- `action="/diagnosticar"` e `method="post"` enviam os sintomas ao FastAPI.
- `name="sintomas"`, `name="pergunta"` e `name="doenca"` precisam corresponder aos parâmetros `Form()` do backend.
- `{% if doenca %}` mostra os recursos de conversa somente quando há classificação disponível.
- `{{ doenca }}`, `{{ prob }}` e `{{ resposta }}` são substituídos pelo Jinja2; o conteúdo textual é escapado, sem usar `|safe`.
- O assistente simples mostra `resposta`, enquanto o agente apresenta `ultima_pergunta` e `ultima_resposta`.
- A classificação exibida é **experimental**, não um diagnóstico clínico. O QR Code aponta para a rede local.

### 7.3. CSS completo — `static/style.css`

**Substitua todo o conteúdo do arquivo pelo código abaixo.** Os comentários `/* ... */` explicam cada grupo de estilos. Mantivemos a identidade visual simples do SaudeIA.

```css
/* ==============================
   RESET BÁSICO
   Remove margens padrão e facilita o cálculo das larguras.
   ============================== */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

/* ==============================
   BODY: aparência geral da página.
   ============================== */
body {
    font-family: Arial, Helvetica, sans-serif;
    background: #f2f5f9;
    padding: 10px;
    color: #263238;
}

/* ==============================
   CONTAINER PRINCIPAL
   Limita a largura para facilitar a leitura.
   ============================== */
.container {
    max-width: 500px;
    width: 100%;
    margin: 40px auto;
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

/* Títulos e separadores das seções. */
h2 {
    text-align: center;
    margin-bottom: 20px;
}
h3 {
    margin-top: 20px;
    margin-bottom: 10px;
}
hr {
    margin-top: 20px;
    border: 0;
    border-top: 1px solid #dce3e9;
}
label {
    display: block;
    margin-top: 12px;
    font-size: 14px;
}

/* ==============================
   TEXTAREA: entrada de texto multilinha.
   ============================== */
textarea {
    width: 100%;
    min-height: 100px;
    padding: 10px;
    margin-top: 10px;
    border-radius: 6px;
    border: 1px solid #ccc;
    resize: vertical;
    font: inherit;
    font-size: 14px;
}

/* ==============================
   BOTÕES E LINK COM APARÊNCIA DE BOTÃO.
   ============================== */
button, .botao {
    display: block;
    margin-top: 12px;
    width: 100%;
    padding: 10px;
    background: #0d6efd;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    text-align: center;
    text-decoration: none;
    cursor: pointer;
}
button:hover, .botao:hover {
    background: #0b5ed7;
}
.botao-secundario {
    background: #6c757d;
}
.botao-secundario:hover {
    background: #5a6268;
}

/* ==============================
   RESULTADO E AVISOS.
   ============================== */
.resultado {
    margin-top: 15px;
    padding: 12px;
    background: #e9ecef;
    border-radius: 6px;
    font-size: 14px;
    overflow-wrap: anywhere;
}
.resultado h2 {
    margin-top: 10px;
    margin-bottom: 10px;
}
.erro {
    color: #a61b1b;
}
.aviso-curto {
    margin-top: 10px;
    font-size: 12px;
    color: #555;
}

/* ==============================
   CHAT: diferencia visualmente usuário e agente.
   ============================== */
.chat-box {
    margin-top: 15px;
}
.chat-user {
    background: #e3f2fd;
    padding: 10px;
    margin: 6px 0;
    border-radius: 10px;
    font-size: 14px;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}
.chat-agent {
    background: #f1f8e9;
    padding: 10px;
    margin: 6px 0;
    border-radius: 10px;
    font-size: 14px;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}

/* ==============================
   AVISO LEGAL E QR CODE.
   ============================== */
.aviso-legal {
    margin-top: 20px;
    font-size: 12px;
    color: #666;
    text-align: center;
}
.qr-code {
    text-align: center;
    margin: 20px 0 15px;
}
.qr-code img, img {
    max-width: 100%;
}
.qr-code p {
    font-size: 12px;
    margin-top: 6px;
}

/* ==============================
   RESPONSIVIDADE: ajustes para celulares.
   ============================== */
@media (max-width: 480px) {
    .container {
        margin: 20px auto;
        padding: 20px;
    }
    textarea {
        min-height: 90px;
    }
    button, .botao {
        font-size: 13px;
    }
}
/* AULA 05: mantém quebras de linha e evita transbordamento de mensagens. */
.chat-user, .chat-agent { overflow-wrap: anywhere; }
.chat-user p, .chat-agent p { margin-top: 7px; white-space: pre-wrap; }
/* Botão secundário separado do envio da pergunta. */
.form-nova-conversa { margin-top: 12px; }
.botao-nova-conversa { background: #526273; }
.botao-nova-conversa:hover { background: #394957; }
/* Avisos de falha e de validação acessíveis. */
.resultado.erro { color: #8a251f; border-left: 4px solid #b9382e; }

/* AULA 05: conversa contínua no mesmo painel. */
.secao-agente { margin-top: 20px; }
.chat-box { max-height: 380px; overflow-y: auto; padding: 10px;
    border: 1px solid #dbe3eb; border-radius: 10px; background: #f8fafc; }
.chat-user, .chat-agent { margin-top: 10px; margin-bottom: 10px; padding: 14px; line-height: 1.5; }
.chat-user { margin-left: 10%; }
.chat-agent { margin-right: 10%; }
.chat-vazio { color: #536476; font-size: 14px; padding: 12px; }
.form-chat { margin-top: 12px; }
.form-chat textarea { min-height: 70px; }
.aviso-conversa { border-left: 4px solid #2d8a66; }
@media (max-width: 480px) {
    .chat-user { margin-left: 5%; }
    .chat-agent { margin-right: 5%; }
}
```

**Entendendo o CSS:** `.container` controla a largura do painel; `textarea` estiliza a entrada de textos; `.resultado` destaca a classificação e as respostas; `.chat-user` e `.chat-agent` diferenciam as mensagens; `@media` adapta o layout a telas menores. Os seletores são usados diretamente no HTML acima.

### 7.4. Verificar os arquivos

Salve os dois arquivos com `Ctrl+S`. A seguir, execute:

```powershell
# Verifica se os arquivos estão no local correto.
Test-Path .\templates\index.html
Test-Path .\static\style.css

# Confere se o Python consegue importar a aplicação.
python -c "import main_api; print('FastAPI importado com sucesso')"
```

**Resultado esperado:** `True`, `True` e a mensagem de importação. O import carrega os modelos salvos da Aula 03; se estiverem ausentes, execute primeiro a etapa de persistência daquela aula.

## 8. Executar e acessar no computador

No terminal do VSCode, execute:

```powershell
python -m uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
```

Espere `Application startup complete`. Abra http://127.0.0.1:8000/ e teste com **dados fictícios**, por exemplo `febre e dor de cabeça`. A previsão é ilustrativa e não corresponde a um diagnóstico clínico.

- `main_api:app`: importa o objeto `app` definido em `main_api.py`.
- `--reload`: reinicia o servidor ao salvar alterações; use somente no desenvolvimento.
- `--host 127.0.0.1`: aceita conexões somente do computador.
- `--port 8000`: utiliza a porta 8000.

## 9. QR Code: abrir no celular pela rede Wi-Fi

O QR Code contém o IP da rede local, por exemplo `http://192.168.0.105:8000`. **Se o servidor escutar apenas em `127.0.0.1`, o celular não conseguirá acessá-lo.**

Pare o servidor com `Ctrl+C` e reinicie:

```powershell
python -m uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
```

`0.0.0.0` significa **escutar nas interfaces de rede disponíveis**; não é o endereço que deve ser digitado no navegador. **Não abra `http://0.0.0.0:8000/`**, pois isso pode gerar `ERR_ADDRESS_INVALID (-108)`.

### 9.1. Qual endereço abrir no navegador?

| Onde acessar                            | Endereço correto               |
| --------------------------------------- | ------------------------------ |
| No próprio computador                   | http://127.0.0.1:8000          |
| Alternativa no computador               | http://localhost:8000          |
| No celular conectado à mesma rede Wi-Fi | `http://IP_DO_COMPUTADOR:8000` |

**Exemplo:** se `ipconfig` mostrar IPv4 `192.168.15.41`, no celular abra `http://192.168.15.41:8000`. O IP do exemplo deve ser substituído pelo IP real do computador. O endereço `0.0.0.0` serve apenas para a configuração de escuta do servidor.

**Procedimento:** (1) conecte computador e celular ao mesmo Wi-Fi; (2) execute `ipconfig` no PowerShell para conferir o IPv4; (3) compare com o endereço exibido em `QR Code criado para: ...`; (4) abra a câmera do celular e leia o QR Code; (5) caso não funcione, verifique se o Firewall do Windows permite Python/Uvicorn em **redes privadas**. Algumas redes isolam os dispositivos e impedem essa conexão. Não abra portas no roteador nem exponha o servidor de desenvolvimento à internet.

**Atenção:** `--reload` é útil apenas para desenvolvimento. O histórico do agente fica na memória do processo e será perdido quando o servidor reiniciar.

## 10. Testar a API JSON

**API x página HTML:** a página web foi feita para pessoas; a API JSON permite que outros programas enviem um texto e recebam um resultado estruturado. A FastAPI gera o Swagger automaticamente a partir das rotas e dos modelos Pydantic.

No navegador, acesse http://127.0.0.1:8000/docs, localize **POST `/api/prever`**, clique em **Try it out** e envie:

```json
{ "texto": "febre e dor de cabeça" }
```

A resposta HTTP **200** terá `classe_prevista`, `confianca_percentual` e `aviso`. O esquema `SaidaPrevisao` declara os três campos para que apareçam tanto no **Schema/Example Value** do Swagger quanto no **Response body** de uma requisição real. A classe e a confiança dependem do modelo treinado.

Exemplo **ilustrativo** (o percentual não é resultado medido):

```json
{
  "classe_prevista": "Febre Maculosa",
  "confianca_percentual": 87.45,
  "aviso": "Classificação exclusivamente didática; não constitui diagnóstico médico."
}
```

A confiança é a estimativa do classificador para a classe prevista, **não** sua acurácia global e **não** a probabilidade clínica de uma doença. Se o Swagger continuar mostrando apenas dois campos, salve `main_api.py`, aguarde o reinício do Uvicorn e atualize `/docs`.

Para testar no PowerShell (em outro terminal):

```powershell
$corpo = @{ texto = "febre e dor de cabeça" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/prever" `
    -Method Post -ContentType "application/json; charset=utf-8" -Body $corpo
```

O classificador local funciona **sem chave de API da OpenAI**. Já os recursos de conversa utilizam a API externa e podem consumir créditos.

**Interpretando os códigos HTTP:** `200` indica processamento bem-sucedido; `422` indica que os dados recebidos não passaram na validação. A confiança percentual expressa a saída do classificador, não a certeza de um diagnóstico.

## 11. Testes automatizados

Crie o arquivo:

```powershell
New-Item -ItemType Directory -Force .\tests | Out-Null
if (-not (Test-Path .\tests\test_api.py)) {
    New-Item -ItemType File .\tests\test_api.py | Out-Null
}
code .\tests\test_api.py
```

**Arquivo:** `tests/test_api.py`. Copie e salve:

```python
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
```

Execute:

```powershell
python -m pytest -q tests/test_api.py
```

**Como ler os testes:** cada função `test_...` verifica uma situação. O `TestClient` simula requisições HTTP, `assert` compara o resultado com o esperado e `monkeypatch` substitui temporariamente a previsão real por um resultado conhecido, evitando que o teste dependa do comportamento estatístico do modelo. Os comentários dentro de `test_api.py` explicam essas operações.

**Resultado esperado:** os testes definidos no arquivo `tests/test_api.py` devem passar, desde que os três arquivos de modelo e os templates estejam presentes. Os testes substituem a previsão por uma resposta simulada, mas ainda precisam importar o módulo `main_api.py`.

### 11.1. Verificação das rotas de navegação

Com o servidor em execução, abra outro terminal PowerShell e execute:

```powershell
# O GET de uma rota originalmente POST deve redirecionar para a página inicial.
curl.exe -s -o NUL -w "%{http_code} %{redirect_url}`n" http://127.0.0.1:8000/diagnosticar
curl.exe -s -o NUL -w "%{http_code} %{redirect_url}`n" http://127.0.0.1:8000/chat

# Verifica a página que apresenta a conversa.
curl.exe -s -o NUL -w "%{http_code}`n" http://127.0.0.1:8000/conversa

# Executa toda a suíte de testes do projeto.
python -m pytest -q tests/
```

Os dois primeiros comandos devem retornar `303` e o endereço da página inicial. O terceiro deve retornar `200`. Para verificar os fluxos de conversa, use o navegador ou os testes automatizados; **não é necessário utilizar a API externa para testar redirecionamentos** quando a resposta do agente é simulada. A quantidade total de testes pode crescer nas aulas seguintes; o critério é que todos sejam aprovados.

## 12. Solução de problemas

| Sintoma                                        | Possível causa e correção                                                         |
| ---------------------------------------------- | --------------------------------------------------------------------------------- |
| `No module named services.agenteIA`            | Crie `services/agenteIA.py` com o código da Seção 4                               |
| `No module named services.diagnosticoIA`       | Confirme o arquivo gerado na Aula 03                                              |
| `FileNotFoundError` em `model/`                | Verifique os três arquivos persistidos na Aula 03                                 |
| `NameError: app is not defined`                | Crie `app = FastAPI()` antes de definir as rotas                                  |
| HTTP 500, `TypeError: unhashable type: 'dict'` | Atualize todas as chamadas de `TemplateResponse()`                                |
| `Form data requires python-multipart`          | Instale `python-multipart` no `.venv`                                             |
| Assistente indisponível                        | Verifique chave, rede, permissões e faturamento da API                            |
| `ERR_ADDRESS_INVALID (-108)`                   | Não abra `0.0.0.0` no navegador; utilize `127.0.0.1` ou o IPv4 real do computador |
| Celular não abre o QR Code                     | Use `--host 0.0.0.0`, confira IP, Wi-Fi e Firewall privado                        |
| `422 Unprocessable Entity`                     | Informe JSON com campo `texto` e ao menos duas palavras                           |

## 13. Git e GitHub

**Por que usar `.gitignore`?** O Git registra arquivos do projeto, mas não deve versionar credenciais, ambientes virtuais, caches ou artefatos gerados automaticamente. O arquivo `.gitignore` define esses padrões. O `requirements.txt`, por outro lado, registra as dependências necessárias para recriar o ambiente.

Antes de publicar, evite versionar credenciais, `.venv`, históricos com dados sensíveis e arquivos de cache. Se ainda não existir, abra `.gitignore` e inclua:

**Crie ou abra o arquivo `.gitignore` na raiz do projeto:**

```powershell
if (-not (Test-Path .\.gitignore)) { New-Item -ItemType File .\.gitignore | Out-Null }
code .\.gitignore
```

Mantenha as regras já existentes e acrescente as ausentes. O conteúdo recomendado é:

```gitignore
# Ambiente e credenciais
.venv/
venv/
.env
.env.*
!.env.example
__pycache__/
*.py[cod]
.pytest_cache/
.vscode/

# Artefatos gerados
model/*.json
model/*.joblib
model/*.pkl
historico.csv
*.log
static/qrcode.png
main_api_backup.py
```

Os arquivos `model/*.json`, `model/*.joblib` e `model/*.pkl` são gerados nas Aulas 02–03. **Não os apague do computador:** o `.gitignore` apenas impede que novos arquivos correspondentes sejam adicionados pelo Git. O `static/qrcode.png` também é regenerado na execução e não precisa ser publicado. O `.env.example` pode ser versionado, mas nunca deve conter uma chave real.

**Importante:** `.gitignore` não remove segredos já enviados anteriormente ao repositório. Se uma chave for publicada, revogue-a e crie outra.

Confira o estado do projeto e registre as alterações:

```powershell
python -m pip freeze > requirements.txt
git status
git add main_api.py services/agenteIA.py templates/index.html static/style.css tests/test_api.py requirements.txt .gitignore
# Se a aula Markdown estiver salva na pasta aulas/, adicione-a também:
# git add aulas/aula-04-aplicacao-web-fastapi.md
git status
git commit -m "Aula 04: FastAPI, agente IA, API e acesso por QR Code"
git push origin main
```

## 14. Checklist final

- [ ] Os três arquivos de modelo da Aula 03 estão disponíveis.
- [ ] `services/agenteIA.py` existe e pode ser importado.
- [ ] A chave de API não aparece no código nem no GitHub.
- [ ] O classificador funciona sem consumir créditos da OpenAI.
- [ ] `/`, `/diagnosticar`, `/conversa`, `/chat_agente`, `/nova_conversa`, `/api/saude`, `/api/prever` e `/docs` funcionam.
- [ ] O Swagger e a resposta real de `/api/prever` apresentam `confianca_percentual`.
- [ ] O `.gitignore` exclui credenciais, artefatos de modelo, caches, QR Code gerado e histórico CSV.
- [ ] Os dois formulários de conversa funcionam quando a API externa está disponível.
- [ ] Copiei e testei os códigos completos de `templates/index.html` e `static/style.css` apresentados na Seção 7.
- [ ] O QR Code abre o site no celular na mesma rede local.
- [ ] Os testes automatizados foram executados.
- [ ] O projeto foi registrado no GitHub.

**Próxima aula:** aperfeiçoamento do agente, testes de regressão, validações e melhorias da aplicação.

---

© 2026 **Wilson Luciano** · Programação para Engenharia Mecânica · DMEC · UFS · SaudeIA — Inteligência Artificial Aplicada à Engenharia · Material didático · Versão 1.0
