from pathlib import Path

import pandas as pd


# =====================================
# Dataset completo carregado do CSV limpo
# =====================================

def dataset_completo():
    """
    Carrega o dataset limpo a partir do arquivo CSV
    e retorna um DataFrame Pandas pronto para vetorização.
    """

    # Arquivo gerado pelo limpar_excel.py
    raiz_projeto = Path(__file__).resolve().parents[1]
    arquivo = raiz_projeto / "dados" / "processados" / "dataset_limpo.csv"

    if not arquivo.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {arquivo}")

    # Lê o CSV
    df = pd.read_csv(arquivo)

    # Segurança: garante que não existem linhas vazias
    df = df.dropna()

    # Converte tudo para string (evita erros no TF-IDF)
    df["sintomas"] = df["sintomas"].astype(str)
    df["diagnostico"] = df["diagnostico"].astype(str)

    return df
