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
