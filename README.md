# Aprendizado Não Supervisionado

Projeto acadêmico do 4º semestre de Biopark.

## Requisitos

Python 3.11 ou 3.12. A implementação usa PyTorch; para reproduzir o ambiente de CPU, instale a distribuição CPU do PyTorch conforme as instruções oficiais e então instale as dependências do projeto.

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Em Linux/macOS, ative o ambiente com `source .venv/bin/activate`.

## Testes

```bash
python -m pytest
```

## Estrutura

- `src/autoencoder/`: código reutilizável do projeto.
- `notebooks/`: narrativa da apresentação; deve importar o pacote de `src/`.
- `tests/`: testes das partes críticas.
- `configs/`: hiperparâmetros e sementes.
- `reports/figures/`: figuras geradas pelos experimentos.
- `app/`: espaço reservado para a demo Streamlit futura.

## Decisões técnicas

- **Framework:** PyTorch. O grupo consegue explicar o fluxo de treino passo a passo, importante para a apresentação, e a API permite expor o espaço latente diretamente. A versão 2.5.1 foi escolhida para Python 3.11/3.12 e mantida fixa; a instalação acima seleciona os wheels CPU.
- **Local:** raiz do repositório, conforme a estrutura definida nesta issue.
- **Reprodutibilidade:** `set_seed()` fixa geradores Python, NumPy e PyTorch. Seeds reduzem variações aleatórias; não garantem resultados idênticos em todo hardware/operação.
