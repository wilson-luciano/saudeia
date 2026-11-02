"""Treina e salva os componentes do SaudeIA."""

from pathlib import Path
import joblib
from services.treinamentoService import treinar_modelo

# Pasta model/ localizada na raiz do projeto.
RAIZ = Path(__file__).resolve().parents[1]
PASTA_MODELO = RAIZ / "model"


def salvar_tudo():
    """Treina e grava XGBoost, TF-IDF e LabelEncoder."""
    # Mantém a ordem de retorno da Aula 02.
    modelo, tfidf, encoder, _, _ = treinar_modelo()

    # Cria a pasta de saída se ainda não existir.
    PASTA_MODELO.mkdir(parents=True, exist_ok=True)

    # O XGBoost possui um método próprio para salvar seu modelo.
    modelo.save_model(str(PASTA_MODELO / "modelo_saudeIA.json"))

    # Salva os objetos do scikit-learn para reutilização.
    joblib.dump(tfidf, PASTA_MODELO / "vetorizador_saudeIA.pkl")
    joblib.dump(encoder, PASTA_MODELO / "encoderY_saudeIA.pkl")

    print("Treinamento concluído. Arquivos salvos em model/:")
    print("- modelo_saudeIA.json")
    print("- vetorizador_saudeIA.pkl")
    print("- encoderY_saudeIA.pkl")


def main():
    """Permite executar este serviço diretamente."""
    salvar_tudo()


if __name__ == "__main__":
    main()
