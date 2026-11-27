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

### Autoencoder denso — issue #4

`Autoencoder.from_config("configs/default.yaml")` carrega `model.latent_dim`
(padrão 16; comparar também 8 e 32). O modelo aceita imagens em lotes `(B, 784)`
ou `(B, 1, 28, 28)` e `forward()` devolve a reconstrução no mesmo formato da
entrada. A arquitetura é
784 → 256 → 64 → latent_dim → 64 → 256 → 784; `encode()` e `decode()` também
podem ser usados separadamente. `count_parameters()` informa o total treinável.

A saída usa Sigmoid para limitar as intensidades reconstruídas ao intervalo
`[0, 1]` das imagens normalizadas. O gargalo é linear, sem ReLU, para preservar
as dimensões latentes com valores positivos e negativos. A função padrão é MSE: ela mede erro quadrático
entre intensidades contínuas. BCE modela cada pixel como Bernoulli e também é
possível para alvos em `[0, 1]`, mas exige essa hipótese probabilística; para este
baseline escolhemos MSE por ser uma comparação direta de intensidade. Um
`latent_dim` maior reduz a compressão e pode permitir reconstruções melhores,
mas deixa o gargalo menos restritivo e pode reduzir a capacidade de detectar
anomalias pelo erro de reconstrução.

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

## Visualizações — issue #7

O notebook usa os checkpoints dos dois cenários (`models/ae_bag.pt` e
`models/ae_sandal.pt`). `models/` não é versionado: treine antes, ou copie os
checkpoints da equipe para essa pasta. Pesos treinados em outra máquina podem
diferir um pouco dos checkpoints usados nos relatórios.

```bash
python scripts/train_cenarios.py --scenario bag
python scripts/train_cenarios.py --scenario sandal
python scripts/run_notebook.py notebooks/03_latent_space.ipynb
```

`run_notebook.py` executa qualquer notebook do zero em um kernel no próprio
processo, como o `run_eda.py`; o Jupyter convencional também funciona.

- `notebooks/03_latent_space.ipynb` (#7): extrai os latentes do teste com `encode()`,
  projeta com PCA e t-SNE (por classe e normal × anomalia) e mede a mistura entre
  classes pelos 10 vizinhos mais próximos nas 16 dimensões, sem distorção de projeção.
  Figuras `latent_*.png`; números em `reports/latent_space_summary.json`.
O código reutilizável fica em `src/autoencoder/inference.py` (carregar checkpoint
e extrair latentes) e `src/autoencoder/visualize.py` (projeções e matriz de vizinhos).

## Estrutura

- `src/autoencoder/`: código reutilizável do projeto.
- `notebooks/`: narrativa da apresentação; deve importar o pacote de `src/`.
- `tests/`: testes das partes críticas.
- `configs/`: hiperparâmetros e sementes.
- `reports/figures/`: figuras geradas pelos experimentos.
- `app/`: espaço reservado para a demo Streamlit futura.

## Reconstruções e falhas — issue #8

`notebooks/04_reconstructions.ipynb` mostra o Autoencoder funcionando e falhando,
com o limiar P95 da validação (decisão da issue #6): grades original | reconstruído |
`|x − x̂|`, melhores e piores reconstruções, falsos positivos e falsos negativos
(os mais extremos e uma amostra sorteada) e a explicação medida de cada tipo de
falha, conectada às hipóteses da EDA. Ele também confere as métricas com
`reports/resultados_deteccao.csv`.

O notebook usa os checkpoints `models/ae_bag.pt` e `models/ae_sandal.pt`. `models/`
não é versionado: treine antes (`python scripts/train_cenarios.py --scenario bag` e
`--scenario sandal`) ou copie os checkpoints da equipe para essa pasta. Pesos
treinados em outra máquina podem diferir um pouco dos usados nos relatórios.

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/04_reconstructions.ipynb --ExecutePreprocessor.timeout=600
```

Figuras `reports/figures/recon_*.png` e `falsos_*.png`; números em
`reports/reconstructions_summary.json`. O código reutilizável fica em
`src/autoencoder/reconstructions.py` (reconstruções por imagem, seleção de
falhas e grades).

## Decisões técnicas

- **Framework:** PyTorch. O grupo consegue explicar o fluxo de treino passo a passo, importante para a apresentação, e a API permite expor o espaço latente diretamente. A versão 2.5.1 foi escolhida para Python 3.11/3.12 e mantida fixa; a instalação acima seleciona os wheels CPU.
- **Local:** raiz do repositório, conforme a estrutura definida nesta issue.
- **Reprodutibilidade:** `set_seed()` fixa geradores Python, NumPy e PyTorch. Seeds reduzem variações aleatórias; não garantem resultados idênticos em todo hardware/operação.
