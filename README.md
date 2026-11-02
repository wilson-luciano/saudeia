# SaudeIA — Inteligência Artificial Aplicada à Engenharia

**Projeto didático da disciplina Programação para Engenharia Mecânica — DMEC · UFS**  
**Professor:** Wilson Luciano · **Material didático · Versão 1.0**

<p align="center">
  <img src="assets/imagens/saudeia-robo.png" alt="SaudeIA — Assistente Inteligente de Saúde" width="360">
</p>

O **SaudeIA** é uma aplicação educacional em Python que demonstra um fluxo completo de aprendizado de máquina aplicado à classificação de textos: preparação de dados, vetorização **TF-IDF**, treinamento e avaliação com **XGBoost**, persistência do modelo, inferência local, interface web com **FastAPI/Jinja2**, API JSON e interação com um modelo de linguagem por meio da API da OpenAI.

Embora os exemplos utilizem descrições fictícias de sintomas, os conceitos são transferíveis a problemas de Engenharia, como classificação de falhas e análise de registros de manutenção.

> **Aviso importante:** o SaudeIA é exclusivamente didático. **Não realiza diagnóstico médico**, não é clinicamente validado e não substitui profissionais de saúde. Use somente dados fictícios. A confiança numérica do classificador **não** representa uma probabilidade clínica validada.

## 1. Funcionalidades

- Tratamento e limpeza de dados tabulares, com verificações de qualidade.
- Representação de textos por TF-IDF, sem ajustar o vetorizador com os dados de teste.
- Treinamento e avaliação de um classificador XGBoost.
- Gravação e carregamento dos artefatos de modelo, vetorizador e codificador de classes.
- Classificação experimental de textos pelo navegador e por API JSON.
- Registro local de classificações em CSV, usando somente exemplos fictícios.
- Assistente de perguntas isoladas e chat contínuo com histórico limitado por sessão.
- Reinício de conversa e redirecionamento **POST/Redirect/GET (HTTP 303)**.
- QR Code para acesso ao servidor na mesma rede local.
- Testes automatizados com `pytest`.

**A API da OpenAI é opcional:** classificação, treinamento, inferência e testes simulados não exigem chave de API. Perguntas reais aos assistentes utilizam um serviço externo, exigem credenciais e podem gerar custos.

## 2. Sequência didática

| Aula | Conteúdo | Arquivo |
|---|---|---|
| 00 | Preparação do ambiente e estrutura do projeto | [`aula-00-preparacao-ambiente-estrutura-projeto-3.md`](aulas/aula-00-preparacao-ambiente-estrutura-projeto-3.md) |
| 01 | Tratamento e limpeza de dados | [`aula-01-tratamento-limpeza-dados.md`](aulas/aula-01-tratamento-limpeza-dados.md) |
| 02 | Vetorização TF-IDF, treinamento e avaliação com XGBoost | [`aula-02-vetorizacao-tfidf-treinamento-xgboost.md`](aulas/aula-02-vetorizacao-tfidf-treinamento-xgboost.md) |
| 03 | Persistência do modelo e previsões | [`aula-03-persistencia-modelo-previsoes.md`](aulas/aula-03-persistencia-modelo-previsoes.md) |
| 04 | Aplicação web com FastAPI, API e QR Code | [`aula-04-aplicacao-web-fastapi.md`](aulas/aula-04-aplicacao-web-fastapi.md) |
| 05 | Aperfeiçoamento e finalização: conversa contínua e testes | [`aula-05-aperfeicoamento-finalizacao-saudeia.md`](aulas/aula-05-aperfeicoamento-finalizacao-saudeia.md) |

As aulas são roteiros práticos para acompanhar a construção do sistema. O repositório também contém a implementação final, permitindo estudar os módulos, executar a aplicação e comparar os resultados.

## 3. Estrutura do projeto

```text
SaudeIA/
├── aulas/                       # Aulas 00 a 05 em Markdown
├── assets/                      # Figuras e fluxogramas didáticos
├── dados/
│   ├── brutos/                  # Dados originais de trabalho
│   └── processados/             # Dados tratados
├── model/                       # Artefatos gerados localmente (ignorados pelo Git)
├── services/                    # Limpeza, treinamento, persistência, inferência e agente
├── static/                      # Estilos, favicon e QR Code gerado
├── templates/                   # Página HTML/Jinja2
├── tests/                       # Testes automatizados
├── main_api.py                  # Aplicação FastAPI
├── requirements.txt             # Dependências Python
├── .env.example                 # Exemplo de variável de ambiente
├── .gitignore                   # Arquivos excluídos do versionamento
├── PLANO_DESENVOLVIMENTO.md     # Planejamento didático
└── README.md
```

Os três arquivos treinados em `model/` são gerados no computador do aluno e não são enviados ao GitHub. O servidor precisa desses arquivos para carregar o classificador.

## 4. Preparação no Windows (PowerShell e VSCode)

**Pré-requisitos:** Python 3.12, VSCode e Git. Os comandos abaixo devem ser executados na raiz do projeto.

Se o projeto já estiver no computador:

```powershell
cd C:\Projetos\SaudeIA
```

Para obter uma cópia do repositório SaudeIA no GitHub:

```powershell
git clone https://github.com/wilson-luciano/saudeia.git
cd SaudeIA
```

Crie e ative um ambiente virtual (somente a criação é necessária na primeira execução):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação de scripts, autorize-a **somente para o processo atual** e tente novamente:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

> **Atenção:** antes de instalar em um ambiente novo, o arquivo `requirements.txt` deve estar salvo em **UTF-8**. A revisão de codificação e compatibilidade das dependências é uma etapa separada da revisão deste README.

## 5. Preparar os dados e gerar o modelo

O classificador é carregado a partir dos artefatos salvos; portanto, **gere o modelo antes de iniciar a aplicação** em uma instalação nova.

```powershell
# Limpeza dos dados e geração do CSV tratado
python -m services.limpeza_dados

# Treinamento, avaliação e apresentação das métricas
python -m services.treinamentoService

# Treinamento e gravação dos artefatos usados na inferência
python -m services.salvarmodelService

# Verificação da inferência local
python -m services.diagnosticoIA
```

A gravação cria:

```text
model/
├── modelo_saudeIA.json
├── vetorizador_saudeIA.pkl
└── encoderY_saudeIA.pkl
```

Esses arquivos **não devem ser adicionados ao Git**. O último comando de persistência executa o treinamento necessário para salvar os artefatos; a execução prévia de `treinamentoService` é útil para observar as métricas na aula.

## 6. Executar a aplicação web

Com o ambiente virtual ativado e os artefatos gerados:

```powershell
python -m uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
```

Abra no navegador:

- **Aplicação:** http://127.0.0.1:8000
- **Documentação interativa da API (Swagger):** http://127.0.0.1:8000/docs
- **Verificação do servidor:** http://127.0.0.1:8000/api/saude

Para testar o QR Code pelo celular na **mesma rede local**, execute:

```powershell
python -m uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
```

O endereço `0.0.0.0` serve para escutar conexões; não é o endereço a digitar no navegador. O QR Code aponta para o IPv4 local detectado pelo programa. O firewall e a configuração da rede podem impedir o acesso pelo celular. **Não exponha o servidor de desenvolvimento diretamente à internet.**

Para encerrar o servidor, pressione `Ctrl+C` no terminal.

## 7. Assistente com OpenAI (opcional)

O assistente simples (`/chat`) responde a perguntas isoladas. O chat contínuo (`/chat_agente`) mantém um histórico limitado em memória RAM, com visualização em `/conversa` e opção de iniciar nova conversa. O histórico **não persiste após reiniciar o servidor** e não oferece autenticação de usuários.

Para utilizar os assistentes, configure uma chave válida na variável de ambiente `OPENAI_API_KEY`. No PowerShell, para a sessão atual:

```powershell
$env:OPENAI_API_KEY = "SUA_CHAVE_AQUI"
```

O arquivo `.env.example` documenta o nome da variável, mas o código atual utiliza `OpenAI()` e **não carrega automaticamente um arquivo `.env`**. Caso opte por armazenar a chave em `.env`, será necessário carregá-la explicitamente ou configurar a variável pelo ambiente de execução.

**Nunca publique chaves, dados reais de pacientes ou conversas privadas no GitHub.** As mensagens enviadas ao assistente são processadas por um serviço externo e podem gerar custos.

## 8. Executar os testes

Com os modelos gerados e o ambiente virtual ativo:

```powershell
# Testes da conversa contínua
python -m pytest -q tests/test_conversas.py

# Testes das rotas de API
python -m pytest -q tests/test_api.py

# Suíte completa
python -m pytest -q tests/
```

Na última execução local informada pelo professor, a suíte completa apresentou **29 testes aprovados**; esse número descreve aquela execução, não uma garantia para outras máquinas ou futuras alterações. Os testes de conversação simulam as respostas externas para não consumir créditos da OpenAI.

## 9. Publicação e boas práticas

Antes de publicar, confira o que será enviado:

```powershell
git status
git ls-files
git diff --check
```

O `.gitignore` exclui ambiente virtual, credenciais, cache, histórico CSV, QR Code gerado e artefatos treinados. **Arquivos já rastreados pelo Git não deixam de ser rastreados apenas porque foram adicionados ao `.gitignore`.** Revise também as permissões de redistribuição dos dados em `dados/` antes de torná-los públicos.

Para alunos: é possível clonar o repositório do professor, desenvolver localmente e publicar uma versão em um repositório próprio, sem permissão de escrita no repositório original.

## 10. Limitações e licença

O modelo é experimental e o conjunto de dados não é clinicamente validado. A classificação não deve orientar decisões médicas. O histórico conversacional fica apenas na memória do processo, sem persistência, expiração automática ou autenticação; a aplicação não está preparada para uso público ou clínico. A avaliação do modelo e seus resultados não devem ser interpretados como validação médica.

O código e a documentação autorais são disponibilizados conforme o arquivo [`LICENSE`](LICENSE). Bibliotecas, imagens e dados de terceiros podem possuir licenças próprias; verifique as respectivas permissões antes de redistribuí-los.

---

© 2026 **Wilson Luciano** · Programação para Engenharia Mecânica · DMEC · UFS · SaudeIA — Inteligência Artificial Aplicada à Engenharia · Material didático · Versão 1.0
