# AGENTS.md

Previsão de indicadores macro do Brasil. O repo tem **duas gerações de pipeline** — saber qual é qual evita editar o lugar errado.

## Onde trabalhar

Pipeline atual (novo):

- `economic_brazil/forecasting/` — baselines, backtest de origem móvel, MASE, conformal e features. `features.py` só usa defasagens e calendário; nada é ajustado ali (sem vazamento por construção).
- `scripts/train.py` — treina **uma** variável: `python -m scripts.train --variable selic`. O `--variable` precisa ter `configs/<variavel>.yaml` correspondente.
- `scripts/gate.py` — lê `artefatos/metrics-*.json`, escreve `artefatos/aprovados.json` e `gate_resumo.md`. Bloqueia MASE ≥ 1 (exceto se o campeão for o próprio random walk) e bloqueia **subcobertura**; cobertura acima do nominal é só aviso, porque o conformal garante cobertura ≥ nominal.
- `scripts/data_quality.py` — valida `dados/economic_data_brazil.pkl`: frescor das 5 metas e mínimo de 40 colunas. Falha alto de propósito.

`codigos_rodando/` é a geração antiga (SARIMAX/CatBoost/RNN, seleção por `Variance > 0.1`). Está fora do ruff, do pylint e do CI. Não use como referência.

Fluxo: `dados.yml` → `dados/economic_data_brazil.pkl` → `data_quality.py` → `train` → `artefatos/metrics-*.json` → `gate` → `aprovados.json` → `publish`.

## Ambiente

- O `.venv` local **não tem pip**: `uv pip install --python .venv\Scripts\python.exe <pacote>`.
- Shell é PowerShell. Use `.\.venv\Scripts\python.exe -m ...`; o `Makefile` usa `python` e serve ao CI (Linux).
- CI roda Python 3.11; o venv local é 3.12. `pyproject.toml` mira 3.11.

```powershell
.\.venv\Scripts\python.exe -m ruff format economic_brazil tests scripts
.\.venv\Scripts\python.exe -m ruff check economic_brazil tests scripts
.\.venv\Scripts\python.exe -m pylint economic_brazil
.\.venv\Scripts\python.exe -m pytest -m "not network and not slow" --timeout=180 -q
```

Teste único: `... -m pytest tests/test_gate.py::test_main_grava_so_os_aprovados -q`.

## Pegadinhas que já quebraram o CI

- `economic_brazil/__init__.py` importa TensorFlow, SHAP, CatBoost e XGBoost com entusiasmo: **qualquer** `import economic_brazil` custa ~5s.
- **Nunca adicione efeito colateral no import.** Um `shap.initjs()` no nível do módulo exigia IPython, que não está no `requirements.txt`, e derrubou coleta e treino em ~5s. O job `import-runtime` (instala só o runtime e roda `import economic_brazil`) existe para pegar essa classe de erro.
- **Nunca chame `input()`** em código de biblioteca: o CI não tem stdin e vira `EOFError`. `set_fred_api_key`/`set_quandl_api_key` levantam `RuntimeError` quando não há terminal.
- **IPEADATA**: use `buscar_serie` de `configuracao_apis/api_ipea.py`, que reaproveita `requests.Session`. Conexão nova custa ~21s de handshake; o `ipeadatapy` abria duas por série e 25 séries estouravam os 600s.
- `dados/economic_data_brazil.pkl` é a fonte única: a coleta grava, `data_quality` valida e `train` lê. Não crie um terceiro nome de arquivo.
- `dados/rodando_coleta_dados_async.py` roda com `working-directory: dados` e faz `sys.path.append('..')`.
- Nos workflows que publicam, a ordem importa: `git add` → `commit` → `pull --rebase --autostash` → `push`. Pull antes do add falha com "cannot pull with rebase: You have unstaged changes".
- Coleta parcial precisa falhar, não publicar menos: o `::warning::` de fontes com falha vira anotação legível no check-run (`/check-runs/<id>/annotations`).

## Testes

- Suíte offline é o padrão: `-m "not network and not slow"`. Marcadores `network` (APIs externas) e `slow` (>30s).
- `tests/test_imports.py` tem um teste em subprocesso que bloqueia IPython de propósito — não remova.
- `tests/test_streamlit.py` é `network` + `slow` e depende do app Streamlit.

## Modelagem (não óbvio)

- Toda variável é avaliada contra 4 baselines; o campeão sai por MASE em origens móveis, nunca no holdout.
- `MASE` usa escala de 1 passo (comparável ao random walk). `MASE_sazonal` existe só como informação — com `season=12` quase todo modelo "bate" o random walk.
- Cobertura conformal é **rolling**: cada erro só vê os `n_calib` anteriores. `n_origens` e `n_calib` nos YAMLs determinam quantos pontos de avaliação existem; abaixo de 20 o gate trata cobertura como aviso.
- `AlvoAncorado` treina em `y - âncora` para árvores não extrapolarem o intervalo visto no treino.

## Dados e artefatos

- `artefatos/metrics-<variavel>.json` são versionados e publicados pelo job `publish` (só os aprovados).
- `gate_resumo.md` e `artefatos/**/*.pkl` são transitórios (gitignored).
- `dados/*.pkl` são versionados; `catboost_info/` foi removido do índice e é regenerável.
- Vários arquivos antigos têm mojibake (latin-1/cp1252). Evite reescrever strings sem necessidade; edite só o trecho alvo.
