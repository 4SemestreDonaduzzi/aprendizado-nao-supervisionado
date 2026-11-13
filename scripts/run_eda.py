"""Executa a EDA do zero em um kernel IPython local, sem abrir portas.

Alternativa ao nbconvert para ambientes que bloqueiam sockets. As saídas,
incluindo figuras e tabelas, são produzidas pelas próprias células e gravadas
somente quando todas terminam sem erro. Execute a partir da raiz do projeto:

    python scripts/run_eda.py
"""

import os
from pathlib import Path
import time

from IPython.utils.capture import capture_output
from ipykernel.inprocess.manager import InProcessKernelManager
import nbformat


def main():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks/01_eda.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    manager = InProcessKernelManager()
    previous_directory = Path.cwd()
    started = time.monotonic()
    completed = 0
    try:
        # Mesmo diretório de trabalho usado ao abrir o arquivo no Jupyter.
        os.chdir(notebook_path.parent)
        manager.start_kernel()
        for index, cell in enumerate(notebook.cells, start=1):
            if cell.cell_type != "code":
                continue
            print(f"Executando célula {index}/{len(notebook.cells)}...", flush=True)
            with capture_output() as captured:
                result = manager.kernel.shell.run_cell(cell.source, store_history=True)
            if not result.success:
                print(captured.stdout)
                print(captured.stderr)
                result.raise_error()
            outputs = []
            if captured.stdout:
                outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            outputs.extend(
                nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata)
                for output in captured.outputs
            )
            cell.outputs = outputs
            cell.execution_count = result.execution_count
            completed += 1
        nbformat.validate(notebook)
        temporary = notebook_path.with_suffix(".ipynb.tmp")
        nbformat.write(notebook, temporary)
        temporary.replace(notebook_path)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel()
        os.chdir(previous_directory)
    print(f"Concluído: {completed} células de código em {time.monotonic() - started:.1f}s.")
    print("Notebook executado salvo em notebooks/01_eda.ipynb.")


if __name__ == "__main__":
    main()
