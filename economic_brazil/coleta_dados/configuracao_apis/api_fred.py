import os
import sys

sys.path.append("..")
from dotenv import load_dotenv, set_key


def set_fred_api_key() -> None:
    """Pede a chave no terminal e grava no .env.

    Em CI nao existe stdin. Sem esta guarda o `input()` levanta EOFError no
    meio da coleta e o job morre com exit 1 sem dizer o motivo.
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "FRED_API_KEY nao definida e nao ha terminal para perguntar. "
            "Defina a variavel de ambiente ou o secret FRED_API_KEY."
        )
    api_key = input("Por favor, insira sua chave de API do FRED: ").strip()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    dotenv_path = os.path.abspath(os.path.join(base_dir, ".env"))

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

    # Adicionar ou atualizar a chave de API no arquivo .env
    set_key(dotenv_path, "FRED_API_KEY", api_key)


# Execute a função se este arquivo for executado como script principal
if __name__ == "__main__":
    set_fred_api_key()
