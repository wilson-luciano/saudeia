# Plano de desenvolvimento — SaudeIA

## Objetivo
Construir passo a passo uma aplicação de ML, com ênfase na metodologia transferível a problemas de Engenharia.

| Aula | Tema | Teste de saída |
|---|---|---|
| 00 | Ambiente e arquitetura | Importação das bibliotecas e pastas |
| 01 | Dados brutos e limpeza | CSV tratado e verificações de qualidade |
| 02 | TF-IDF, XGBoost e avaliação | Treino/teste sem vazamento, métricas |
| 03 | Predição e persistência | Modelo salvo, carregado e testado |
| 04 | API FastAPI | Endpoint de classificação responde |
| 05 | Interface web | Formulário e resultado no navegador |
| 06 | Histórico opcional, QR Code e chat | Recursos integrados e erros tratados |
| 07 | Testes finais, README e publicação | Execução reproduzível e documentação |

## Critérios técnicos
- Manter o dataset original intocado.
- Documentar faltantes, duplicatas e classes; não descartar anomalias automaticamente.
- Separar treino/teste antes de ajustar transformadores.
- Evitar registros idênticos em treino e teste.
- Não interpretar probabilidades do classificador como confiança clínica.
- Histórico de sintomas desativado por padrão; exemplos fictícios.
- API OpenAI opcional e segredos fora do Git.
- Cada aula: objetivo, teoria mínima, implementação, teste e commit.
- Imagens didáticas em `assets/`, conforme necessidade.

## Observação
Os arquivos de `services/` e `main_api.py` serão criados progressivamente; a estrutura é definitiva, mas não constitui um programa completo nesta etapa.
