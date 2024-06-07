# https://docs.streamlit.io/develop/api-reference/app-testing
# sys.path.append("..")
import os

os.chdir("..")
from streamlit.testing.v1 import AppTest
import warnings
import pytest

warnings.filterwarnings("ignore")

# pylint: disable=W0621
@pytest.fixture(scope="module")
def app_test():
    # Retorna uma instância de AppTest apontando para o script principal do Streamlit
    return AppTest.from_file(
        "codigos_rodando/avaliacao_modelos/apresentacao_streamlit/streamlit_resultados.py",
        default_timeout=50,
    ).run()
# pylint: disable=W0621

def test_streamlit_app_rodando(app_test):
    # Testa se o app foi iniciado corretamente
    assert app_test is not None
