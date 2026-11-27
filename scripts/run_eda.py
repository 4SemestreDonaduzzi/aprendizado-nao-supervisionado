"""Executa a EDA do zero em um kernel IPython local, sem abrir portas.

Atalho para ``python scripts/run_notebook.py notebooks/01_eda.ipynb``. Execute
a partir da raiz do projeto:

    python scripts/run_eda.py
"""

from pathlib import Path

from run_notebook import run_notebook

if __name__ == "__main__":
    run_notebook(Path(__file__).resolve().parents[1] / "notebooks/01_eda.ipynb")
