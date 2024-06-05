# write test for all imports


def test_imports_economic_brazil():
    # pylint: disable=unused-import
    from economic_brazil.coleta_dados.economic_data_brazil import data_economic

    # pylint: disable=unused-import


def test_data_processing():
    # pylint: disable=unused-import
    from economic_brazil.processando_dados.data_processing import (
        criando_dummy_covid,
        criando_defasagens,
        criando_mes_ano_dia,
        escalando_dados,
    )

    # pylint: disable=unused-import


def test_divisao_treino_teste():
    # pylint: disable=unused-import
    from economic_brazil.processando_dados.divisao_treino_teste import (
        treino_test_dados,
        treino_teste_seies_temporal,
    )

    # pylint: disable=unused-import


def test_estacionaridade():
    # pylint: disable=unused-import
    from economic_brazil.processando_dados.estacionaridade import Estacionaridade

    # pylint: disable=unused-import


def test_arima_treinamento():
    # pylint: disable=unused-import
    from economic_brazil.treinamento.arima_treinamento import Arima

    # pylint: disable=unused-import


def test_modelos_treinamento():
    # pylint: disable=unused-import
    from economic_brazil.treinamento.modelos_treinamento import TreinamentoModelos

    # pylint: disable=unused-import


def test_redes_neurais_treinamento():
    # pylint: disable=unused-import
    from economic_brazil.treinamento.redes_neurais_recorrentes import RnnModel

    # pylint: disable=unused-import


def test_treinamento_modelos_tuning():
    # pylint: disable=unused-import
    from economic_brazil.treinamento.treinamento_modelos_tuning import (
        TimeSeriesModelTuner,
    )

    # pylint: disable=unused-import


def test_codigos_graficos():
    # pylint: disable=unused-import
    from economic_brazil.visualizacoes_graficas.codigos_graficos import Graficos

    # pylint: disable=unused-import


def test_analise_modelos_regressao():
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


def test_regressao_conformal():
    # pylint: disable=unused-import
    from economic_brazil.analisando_modelos.regressao_conformal import (
        ConformalRegressionPlotter,
        ConformalAvaliandoMetodo,
    )

    # pylint: disable=unused-import


def test_treinamento_algoritmos():
    # pylint: disable=unused-import
    from economic_brazil.treinamento.treinamento_algoritimos import (
        carregar,
        TreinandoModelos,
    )

    # pylint: disable=unused-import


def test_tratando_dados():
    # pylint: disable=unused-import
    from economic_brazil.processando_dados.tratando_dados import TratandoDados

    # pylint: disable=unused-import
