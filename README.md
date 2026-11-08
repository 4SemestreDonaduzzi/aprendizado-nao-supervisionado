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

Os testes unitários não acessam a internet. Os dois testes de integração usam os
arquivos oficiais já presentes em `data/`; se o cache ainda não existir, aparecem
como `skipped`. Para validar também os dados reais, execute o notebook abaixo e
rode os testes novamente. Para rodar somente os testes offline:

```bash
python -m pytest -m "not integration" -q
```

## Dados e análise exploratória — issues #2 e #3

O notebook baixa os quatro arquivos oficiais do Fashion-MNIST para `data/`,
confere os checksums da fonte e reutiliza arquivos íntegros nas próximas execuções.
Um arquivo corrompido gera erro com orientação para removê-lo e baixar novamente.

Com o ambiente virtual ativado, execute na raiz do repositório:

```bash
python -m ipykernel install --sys-prefix --name python3 --display-name "Python (Fashion-MNIST)"
python -m jupyterlab notebooks/01_eda.ipynb
```

Selecione o kernel **Python (Fashion-MNIST)** e use **Restart Kernel and Run All Cells**.
O notebook também pode ser executado inteiro pelo terminal:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb --ExecutePreprocessor.timeout=300
python -m pytest -q
```

Se o ambiente bloquear a abertura de portas pelo kernel Jupyter, há uma execução
alternativa com um kernel IPython no próprio processo, também iniciado do zero:

```bash
python scripts/run_eda.py
```

Ela executa todas as células em ordem, falha se qualquer uma tiver erro e grava
as saídas reais no notebook. Não exige abrir o Jupyter no navegador.

Saídas da análise:

- `notebooks/01_eda.ipynb`: explicações, verificações e figuras incorporadas.
- `reports/figures/`: seis figuras, incluindo as distribuições dos dois cenários.
- `reports/eda_summary.json`: contagens, estatísticas, hashes dos arquivos e das divisões.
- `reports/validacao_dados_eda.md`: registro da validação e orientação para a apresentação.

A EDA visual usa somente o **treino oficial**. No teste, verificamos integridade,
contagens e o funcionamento do pipeline, sem consultar imagens ou resultados de
detecção para tomar decisões. As hipóteses estão registradas antes do treino do
Autoencoder; ainda não há resultado de desempenho do modelo nesta entrega.

### Configuração e contrato para o treinamento

`configs/default.yaml` define seed, classe anômala, fração de validação, tamanho do
lote e número de workers. `anomaly_class: 8` seleciona Bag; `5` seleciona Sandal.
O notebook explora os dois cenários, independentemente da classe padrão escolhida.
`data.directory` é relativo à raiz do projeto, não à pasta em que o notebook foi aberto.

```python
from autoencoder.data import load_config, load_splits, make_dataloaders

config = load_config("configs/default.yaml")
splits = load_splits(config)
loaders = make_dataloaders(splits, config.batch_size, config.num_workers)
images, original_labels = next(iter(loaders["train"]))
# images: float32, (N, 1, 28, 28), valores entre 0 e 1.
# original_labels: int64, classes originais 0..9; não são o alvo da reconstrução.
# Em uma rede densa: inputs = images.flatten(1).
# A perda compara a reconstrução com a própria entrada.
test_is_anomaly = (splits.test.labels == config.anomaly_class).astype("int64")
```

Cada `DataSplit` contém `images`, `labels`, `indices` e `source`. Os índices
identificam registros dentro do arquivo indicado por `source`; números iguais
em arquivos oficiais diferentes não significam compartilhamento de amostras.
Para os padrões de 15% de validação e uma classe anômala:

| Conjunto | Origem | Normais | Anômalas |
|---|---|---:|---:|
| Treino | Treino oficial | 45.900 | 0 |
| Validação | Treino oficial | 8.100 | 0 |
| Teste | Teste oficial | 9.000 | 1.000 |

As 6.000 imagens anômalas do treino oficial são excluídas dos conjuntos de ajuste.
O loader de treino embaralha com um gerador local de seed fixa; validação e teste
preservam a ordem. Nenhum loader descarta o último lote.

O limiar de detecção deve ser estimado **somente com erros de validação normal**.
O teste fica reservado para a avaliação final após congelar as decisões.

Fonte: [Fashion-MNIST — Zalando Research](https://github.com/zalandoresearch/fashion-mnist).

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
