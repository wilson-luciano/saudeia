"""Testes rápidos da Aula 02, sem treinar o modelo completo."""

from pathlib import Path
import pandas as pd
import pytest
from services.treinamento_ml import carregar_dados


def test_csv_valido(tmp_path: Path):
    # Monta um CSV mínimo com duas classes e dois exemplos de cada.
    caminho = tmp_path / "exemplo.csv"
    pd.DataFrame({
        "sintomas": ["texto a", "texto b", "texto c", "texto d"],
        "diagnostico": ["classe_1", "classe_1", "classe_2", "classe_2"],
    }).to_csv(caminho, index=False)
    dados = carregar_dados(caminho)
    assert len(dados) == 4
    assert dados["diagnostico"].nunique() == 2


def test_coluna_ausente(tmp_path: Path):
    # Um arquivo sem o rótulo não pode alimentar o treinamento.
    caminho = tmp_path / "incompleto.csv"
    pd.DataFrame({"sintomas": ["texto a"]}).to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="Colunas ausentes"):
        carregar_dados(caminho)


def test_classe_com_um_exemplo(tmp_path: Path):
    # Evita uma divisão estratificada impossível de executar.
    caminho = tmp_path / "raro.csv"
    pd.DataFrame({
        "sintomas": ["a", "b", "c"],
        "diagnostico": ["classe_1", "classe_1", "classe_2"],
    }).to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="pelo menos dois"):
        carregar_dados(caminho)
