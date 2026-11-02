# SaudeIA — Projeto didático de Machine Learning

Projeto de ensino com construção progressiva de um classificador experimental de textos de sintomas. O fluxo de tratamento de dados e ML também será discutido para aplicações de Engenharia, como vibração e classificação de falhas.

**Aviso:** ferramenta exclusivamente educacional. Não é dispositivo médico, não fornece diagnóstico clínico nem substitui profissionais de saúde. O conjunto de dados não é clinicamente validado.

## Organização

- `aulas/`: roteiros práticos (a produzir, uma aula por arquivo).
- `dados/brutos/`: fonte original preservada.
- `dados/processados/`: exemplo de saída limpa para conferência; será reproduzida pelos alunos.
- `services/`: módulos Python implementados progressivamente.
- `main_api.py`: aplicação web, implementada nas aulas finais.
- `templates/` e `static/`: recursos da interface web já disponíveis para estudo e adaptação.
- `assets/`: imagens didáticas usadas nas aulas e slides.
- `model/`: modelos gerados localmente, não versionados.
- `tests/`: testes progressivos.

## Metodologia

O aluno clona o repositório do professor e desenvolve localmente. Pode publicar em seu GitHub pessoal, configurando seu próprio remoto `origin`. Não é necessário ter permissão de escrita no repositório do professor.

Os módulos de implementação não são fornecidos prontos: são criados nas aulas. Os arquivos de interface são materiais iniciais a adaptar, não uma aplicação executável completa.

## Licença

Código e documentação autorais disponibilizados sob MIT, conforme `LICENSE`. Bibliotecas e recursos de terceiros mantêm suas próprias licenças. Antes de redistribuir os dados de origem, conferir a proveniência e as permissões de uso.
