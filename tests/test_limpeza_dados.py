from services.limpeza_dados import normalizar_rotulo, normalizar_texto, preparar_dados


def test_normalizar_texto():
    assert normalizar_texto("  Vibração,   ALTA! ") == "vibracao alta"


def test_normalizar_rotulo():
    assert normalizar_rotulo("Falha Mecânica") == "falha_mecanica"


def test_dataset_tratado():
    dados, indicadores = preparar_dados()
    assert list(dados.columns) == ["sintomas", "diagnostico"]
    assert not dados.isna().any().any()
    assert not dados.duplicated().any()
    assert (dados["sintomas"].str.len() > 0).all()
    assert indicadores["registros_finais"] == len(dados)