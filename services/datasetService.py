"""Leitura e verificação dos dados para treinamento do SaudeIA."""
from pathlib import Path
import pandas as pd

# Pasta raiz do projeto: C:\Projetos\SaudeIA.
RAIZ = Path(__file__).resolve().parents[1]


def dataset_completo():
    """Devolve um DataFrame com sintomas e diagnóstico, sem duplicatas exatas."""
    # Prioriza o CSV tratado na Aula 01; aceita a localização original.
    candidatos = [
        RAIZ / "dados" / "processados" / "dataset_limpo.csv",
        RAIZ / "dataset_limpo.csv",
    ]
    caminho = next((p for p in candidatos if p.is_file()), None)
    if caminho is None:
        raise FileNotFoundError("dataset_limpo.csv não encontrado no projeto")

    df = pd.read_csv(caminho, encoding="utf-8")
    if not {"sintomas", "diagnostico"}.issubset(df.columns):
        raise ValueError("O CSV precisa ter as colunas sintomas e diagnostico")

    df = df[["sintomas", "diagnostico"]].dropna().copy()
    df["sintomas"] = df["sintomas"].astype(str).str.strip()
    df["diagnostico"] = df["diagnostico"].astype(str).str.strip()
    df = df[(df["sintomas"] != "") & (df["diagnostico"] != "")]
    df = df.drop_duplicates(subset=["sintomas", "diagnostico"])

    # Uma frase idêntica não deve corresponder a rótulos conflitantes.
    if df["sintomas"].duplicated().any():
        raise ValueError("Há sintomas idênticos com diagnósticos diferentes")
    if df["diagnostico"].nunique() < 2:
        raise ValueError("O treinamento requer pelo menos duas classes")
    return df.reset_index(drop=True)


def main():
    df = dataset_completo()
    print("Registros disponíveis:", len(df))
    print("Classes:", df["diagnostico"].nunique())
    print(df.head())


if __name__ == "__main__":
    main()
