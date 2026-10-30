# https://docs.streamlit.io/develop/api-reference/app-testing
# sys.path.append("..")
import os
import warnings

import pytest
from streamlit.testing.v1 import AppTest

warnings.filterwarnings("ignore")

pytestmark = [pytest.mark.network, pytest.mark.slow]
path = os.path.abspath("../Predicoes_macroeconomicas")


# pylint: disable=W0621
@pytest.fixture(scope="module")
def app_test() -> AppTest:
    # Retorna uma instância de AppTest apontando para o script principal do Streamlit
    return AppTest.from_file(
        path
        + "/codigos_rodando/avaliacao_modelos/apresentacao_streamlit/1_Predicoes-Macroenomicas.py",
        default_timeout=100,
    ).run()


# pylint: disable=W0621


def test_streamlit_app_rodando(app_test: AppTest) -> None:
    # Testa se o app foi iniciado corretamente
    assert app_test is not None
