# write test for all imports


def test_imports_economic_brazil() -> None:
    # pylint: disable=unused-import:
    # pylint: disable=unused-import
    from economic_brazil import EconomicBrazil

    # pylint: disable=unused-import


def test_data_processing() -> None:
    # pylint: disable=unused-import
    from economic_brazil import (
        criando_dummy_covid,
        criando_defasagens,
        criando_mes_ano_dia,
        escalando_dados,
    )

    # pylint: disable=unused-import


def test_divisao_treino_teste() -> None:
    # pylint: disable=unused-import
    from economic_brazil import (
        treino_test_dados,
        treino_teste_seies_temporal,
    )

    # pylint: disable=unused-import


def test_estacionaridade() -> None:
    # pylint: disable=unused-import
    from economic_brazil import Estacionaridade

    # pylint: disable=unused-import


def test_arima_treinamento() -> None:
    # pylint: disable=unused-import
    from economic_brazil import Arima

    # pylint: disable=unused-import


def test_modelos_treinamento() -> None:
    # pylint: disable=unused-import
    from economic_brazil import TreinamentoModelos

    # pylint: disable=unused-import


def test_redes_neurais_treinamento() -> None:
    # pylint: disable=unused-import
    from economic_brazil import RnnModel

    # pylint: disable=unused-import


def test_treinamento_modelos_tuning() -> None:
    # pylint: disable=unused-import
    from economic_brazil import (
        TimeSeriesModelTuner,
    )

    # pylint: disable=unused-import


def test_analise_modelos_regressao() -> None:
    # pylint: disable=unused-import
    from economic_brazil.analisando_modelos.analise_modelos_regressao import (
        MetricasModelos,
        MetricasModelosDicionario,
        PredicaosModelos,
    )

    # pylint: disable=unused-import


def test_importancia_caracteristicas():
    # pylint: disable=unused-import
    from economic_brazil.analisando_modelos.importancia_carecteristicas import (
        ImportanciaRandomForest,
        ImportanciaShap,
    )

    # pylint: disable=unused-import


def test_regressao_conformal() -> None:
    # pylint: disable=unused-import
    from economic_brazil.analisando_modelos.regressao_conformal import (
        ConformalRegressionPlotter,
        ConformalAvaliandoMetodo,
    )

    # pylint: disable=unused-import


def test_treinamento_algoritmos() -> None:
    # pylint: disable=unused-import
    from economic_brazil.treinamento.treinamento_algoritimos import (
        carregar,
        TreinandoModelos,
    )

    # pylint: disable=unused-import


def test_tratando_dados() -> None:
    # pylint: disable=unused-import
    from economic_brazil.processando_dados.tratando_dados import TratandoDados

    # pylint: disable=unused-import
