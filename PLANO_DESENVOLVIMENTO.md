# Plano de Desenvolvimento — SaudeIA

**Disciplina:** Programação para Engenharia Mecânica · **Projeto:** SaudeIA  
**Situação:** sequência didática concluída (Aulas 00 a 05) · **Versão:** 1.0

> **Finalidade exclusivamente didática.** O SaudeIA demonstra técnicas de processamento de linguagem natural, aprendizado de máquina e desenvolvimento web. Suas classificações não constituem diagnósticos médicos; os dados e resultados não possuem validação clínica. Nas atividades, utilize somente exemplos fictícios.

## 1. Objetivo

Desenvolver, de forma incremental, uma aplicação educacional que recebe descrições textuais fictícias, prepara os dados, representa textos por TF-IDF, treina e avalia um classificador XGBoost, persiste os artefatos e disponibiliza uma interface web com API e recursos de conversação. A metodologia pode ser discutida em outros problemas de Engenharia, como classificação de falhas e análise de registros de manutenção.

## 2. Sequência definitiva de aulas

| Aula                                                           | Conteúdo principal                                                                          | Entrega ou verificação                                                     |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [00](aulas/aula-00-preparacao-ambiente-estrutura-projeto-3.md) | Preparação do ambiente, VSCode, PowerShell, Python, ambiente virtual e estrutura do projeto | Ambiente configurado e dependências disponíveis                            |
| [01](aulas/aula-01-tratamento-limpeza-dados.md)                | Inspeção, tratamento e limpeza de dados textuais                                            | Dados preparados e verificações de qualidade                               |
| [02](aulas/aula-02-vetorizacao-tfidf-treinamento-xgboost.md)   | Vetorização TF-IDF, treinamento XGBoost e avaliação                                         | Classificador treinado e métricas de teste                                 |
| [03](aulas/aula-03-persistencia-modelo-previsoes.md)           | Persistência dos artefatos e previsão com o modelo carregado                                | Modelo, vetorizador e codificador de classes salvos e reutilizados         |
| [04](aulas/aula-04-aplicacao-web-fastapi.md)                   | Aplicação FastAPI, interface HTML/CSS, API JSON, QR Code e assistente simples               | Aplicação acessível no navegador e rotas testadas                          |
| [05](aulas/aula-05-aperfeicoamento-finalizacao-saudeia.md)     | Conversa contínua, histórico em sessão, reinício de conversa, testes e Git/GitHub           | Interface final, fluxo POST/Redirect/GET e validação do conjunto de testes |

**Escopo concluído na Aula 05:** não há Aulas 06 e 07 nesta sequência. As atividades de histórico, QR Code, testes finais e publicação foram incorporadas às Aulas 04 e 05.

## 3. Arquitetura do projeto

- **`dados/`** — dados brutos e processados utilizados nas etapas de preparação e treinamento.
- **`services/`** — módulos de limpeza, vetorização, treinamento, persistência, inferência e integração com o modelo de linguagem.
- **`model/`** — artefatos gerados localmente pelo treinamento, necessários à inferência; não são incluídos no Git conforme a configuração atual.
- **`main_api.py`** — aplicação FastAPI e suas rotas.
- **`templates/` e `static/`** — interface HTML, estilos e recursos estáticos.
- **`tests/`** — testes automatizados dos componentes e das rotas.
- **`aulas/` e `assets/`** — roteiros didáticos, figuras e fluxogramas.

O classificador **TF-IDF + XGBoost** realiza inferência local. As funcionalidades de conversação que utilizam a **API da OpenAI** são separadas e dependem de credenciais, conexão e disponibilidade do serviço externo.

## 4. Critérios técnicos e didáticos

1. Preservar a fonte original e documentar as operações de limpeza, os registros ausentes, as duplicatas e a distribuição das classes.
2. Separar treino e teste antes do ajuste de transformadores, reduzindo o risco de vazamento de dados.
3. Diferenciar a **acurácia de avaliação** da **confiança estimada para uma previsão**; nenhuma delas equivale a validade clínica.
4. Manter a lógica de inferência e de conversação em componentes identificáveis, com tratamento de entradas inválidas e falhas externas.
5. Limitar o histórico de conversa em memória, sem tratá-lo como armazenamento persistente ou mecanismo de autenticação.
6. Não publicar credenciais, arquivos `.env`, histórico de uso ou artefatos gerados localmente que estejam excluídos pelo `.gitignore`.
7. Utilizar exemplos fictícios, evitando dados pessoais ou de pacientes.
8. Em cada aula, priorizar objetivo, teoria estritamente necessária, implementação completa, validação e comandos de Git/GitHub.

## 5. Validação e execução

No PowerShell, a partir da pasta do projeto e com o ambiente virtual ativo:

```powershell
cd C:\Projetos\SaudeIA
.\.venv\Scripts\Activate.ps1
python -m pytest -q tests/
python -m uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
```

Abra `http://127.0.0.1:8000` para utilizar a interface ou `http://127.0.0.1:8000/docs` para consultar a documentação interativa da API. Os três artefatos do classificador precisam ter sido gerados conforme as Aulas 02 e 03 antes da inicialização da aplicação.

**Validação informada pelo professor:** a execução local da suíte completa apresentou **29 testes aprovados**. Esse resultado descreve a execução realizada, não substitui a verificação em outro computador ou em um ambiente recém-instalado.

## 6. Publicação e manutenção

A versão publicada no GitHub deve refletir as seis aulas e os arquivos efetivamente utilizados. Antes de publicar, verificar o `git status`, revisar as exclusões de arquivos antigos, conferir o `.gitignore` e executar novamente os testes. Arquivos gerados automaticamente, ambientes virtuais, credenciais e cópias redundantes não devem integrar a versão final.

A publicação do código não dispensa a conferência das licenças e permissões de redistribuição dos conjuntos de dados e de outros recursos de terceiros.

---

© 2026 **Wilson Luciano** · Programação para Engenharia Mecânica · DMEC · UFS · SaudeIA — Inteligência Artificial Aplicada à Engenharia · Material didático · Versão 1.0
