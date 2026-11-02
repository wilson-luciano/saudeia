"""Carregamento dos modelos e inferência didática do SaudeIA."""

from pathlib import Path
import joblib
import xgboost as xgb


class DiagnosticoIA:
    """Reutiliza os três componentes produzidos no treinamento."""

    # Apresentação amigável das classes do conjunto de demonstração.
    MAPA_DOENCAS = {
        "anemia_falciforme": "Anemia Falciforme",
        "artrite_reumatoide": "Artrite Reumatoide",
        "diabetes_tipo1": "Diabetes Tipo 1",
        "doenca_alzheimer": "Doença de Alzheimer",
        "doenca_crohn": "Doença de Crohn",
        "doenca_lyme": "Doença de Lyme",
        "doenca_parkinson": "Doença de Parkinson",
        "doenca_wilson": "Doença de Wilson",
        "esclerose_lateral": "Esclerose Lateral",
        "esclerose_multipla": "Esclerose Múltipla",
        "febre_maculosa": "Febre Maculosa",
        "fibromialgia": "Fibromialgia",
        "hipertireoidismo": "Hipertireoidismo",
        "lupus": "Lúpus",
        "miastenia_gravis": "Miastenia Gravis",
        "porfiria": "Porfiria",
        "sarcoidose": "Sarcoidose",
        "sindrome_fadiga_cronica": "Síndrome da Fadiga Crônica",
        "sindrome_sjogren": "Síndrome de Sjögren",
    }

    def __init__(self, pasta_model="model"):
        """Carrega modelo, vetorizador e encoder da pasta indicada."""
        pasta = Path(pasta_model)
        if not pasta.is_absolute():
            # Resolve caminhos relativos a partir da raiz do projeto.
            pasta = Path(__file__).resolve().parents[1] / pasta

        arquivos = [
            pasta / "modelo_saudeIA.json",
            pasta / "vetorizador_saudeIA.pkl",
            pasta / "encoderY_saudeIA.pkl",
        ]
        faltantes = [p.name for p in arquivos if not p.is_file()]
        if faltantes:
            raise FileNotFoundError(
                "Arquivos do modelo ausentes: " + ", ".join(faltantes)
                + ". Execute: python -m services.salvarmodelService"
            )

        # XGBoost: carrega o arquivo JSON criado por save_model().
        self.modelo = xgb.XGBClassifier()
        self.modelo.load_model(str(arquivos[0]))

        # scikit-learn: carrega objetos previamente serializados.
        # Nunca carregue arquivos .pkl de origem desconhecida.
        self.vectorizer = joblib.load(arquivos[1])
        self.encoder = joblib.load(arquivos[2])

    def prever(self, texto):
        """Retorna (classe_legivel, confianca) para um texto novo."""
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError("Informe um texto não vazio.")

        # Não reaprende o vocabulário: usa SOMENTE transform().
        X = self.vectorizer.transform([texto.strip()])
        classe = self.modelo.predict(X)[0]
        codigo = self.encoder.inverse_transform([int(classe)])[0]

        # Apresenta o nome de forma amigável, sem mudar a classe.
        nome = str(codigo)
        nome_legivel = self.MAPA_DOENCAS.get(
            nome, nome.replace("_", " ").title()
        )

        # Confiança do classificador, não probabilidade clínica.
        probabilidades = self.modelo.predict_proba(X)[0]
        confianca = float(max(probabilidades))
        return nome_legivel, confianca


def main():
    """Teste simples de previsão pelo terminal."""
    ia = DiagnosticoIA("model")
    texto = "febre, dor de cabeça e cansaço"
    classe, confianca = ia.prever(texto)
    print("Sintomas:", texto)
    print("Classe prevista:", classe)
    print("Confiança do classificador:", f"{confianca:.2%}")
    print("AVISO: demonstração didática; não é diagnóstico médico.")


if __name__ == "__main__":
    main()
