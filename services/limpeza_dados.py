"""Aula 01 — limpeza reproduzível do dataset didático SaudeIA."""

from pathlib import Path
import re
import unicodedata

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
ORIGEM = RAIZ / "dados" / "brutos" / "dados_originais.csv"
DESTINO = RAIZ / "dados" / "processados" / "dataset_limpo.csv"


def normalizar_texto(valor: str) -> str:
    """Padroniza caixa, acentos, pontuação e espaços."""
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower().strip()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_rotulo(valor: str) -> str:
    """Transforma o rótulo em identificador padronizado."""
    return normalizar_texto(valor).replace(" ", "_")


def preparar_dados(origem: Path = ORIGEM) -> tuple[pd.DataFrame, dict]:
    """Lê, valida, limpa e retorna dados e indicadores de qualidade."""
    if not origem.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {origem}")

    bruto = pd.read_csv(origem, encoding="utf-8")
    colunas = {"sintomas_limpo", "diagnostico_limpo"}
    faltantes = colunas - set(bruto.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes: {sorted(faltantes)}")

    dados = bruto[["sintomas_limpo", "diagnostico_limpo"]].copy()
    dados.columns = ["sintomas", "diagnostico"]
    total_inicial = len(dados)

    # Retira ausentes antes de converter valores para string.
    dados = dados.dropna(subset=["sintomas", "diagnostico"])
    dados["sintomas"] = dados["sintomas"].map(normalizar_texto)
    dados["diagnostico"] = dados["diagnostico"].map(normalizar_rotulo)
    dados = dados[(dados["sintomas"] != "") & (dados["diagnostico"] != "")]
    apos_validacao = len(dados)

    # Um mesmo texto com rótulos diferentes é ambíguo: não decidir no escuro.
    ambiguos = dados.groupby("sintomas")["diagnostico"].nunique()
    quantidade_ambiguos = int((ambiguos > 1).sum())
    if quantidade_ambiguos:
        exemplos = ambiguos[ambiguos > 1].index.tolist()[:3]
        raise ValueError(
            f"Encontrados {quantidade_ambiguos} textos com rótulos conflitantes. "
            f"Revise os dados. Exemplos: {exemplos}"
        )

    antes_duplicatas = len(dados)
    dados = dados.drop_duplicates(subset=["sintomas", "diagnostico"])
    dados = dados.reset_index(drop=True)

    indicadores = {
        "registros_brutos": total_inicial,
        "removidos_ausentes_ou_vazios": total_inicial - apos_validacao,
        "duplicatas_removidas": antes_duplicatas - len(dados),
        "registros_finais": len(dados),
        "classes": dados["diagnostico"].nunique(),
    }
    return dados, indicadores


def main() -> None:
    dados, indicadores = preparar_dados()
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    dados.to_csv(DESTINO, index=False, encoding="utf-8")

    print("=== RELATÓRIO DE QUALIDADE — SAUDEIA ===")
    for nome, valor in indicadores.items():
        print(f"{nome}: {valor}")
    print(f"Arquivo gerado: {DESTINO}")
    print("\nAmostra tratada:")
    print(dados.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
