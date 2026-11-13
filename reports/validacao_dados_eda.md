# Validação — dados e EDA do Fashion-MNIST

Entrega das issues #2 e #3 de Rafael, integrada à estrutura da `main` no commit
`212f1b0a983945d7f9570c63e78ed53bd84a370f`. Esta entrega prepara os dados e a
análise exploratória; não contém treinamento nem resultados de detecção.

## O que foi verificado

| Verificação | Resultado observado |
|---|---|
| Testes offline, incluindo o teste de seed já existente | 30 passaram |
| Integração com os arquivos oficiais, Bag e Sandal | 2 passaram |
| Total de testes | **32 passaram, sem skips**, com o cache real presente |
| Integridade dos quatro arquivos baixados | MD5 iguais aos publicados pela Zalando |
| Treino e validação | Zero registros compartilhados, cobertura exata dos normais |
| Classe anômala no treino/validação | Zero, nos dois cenários |
| Normalização | `float32`, valores finitos entre 0 e 1 |
| Mesma seed | Mesmos índices e arrays; ordem dos loaders também testada |
| Conjunto de teste | Mantido na ordem oficial, com as dez classes |
| DataLoaders | `(N, 1, 28, 28)`, rótulos `int64`, sem descarte de exemplos |
| Notebook | 8 células de código executadas em ordem, a partir de kernel novo |
| Figuras | 6 PNGs gerados e revisados visualmente |
| Dependências | `pip check` sem dependências quebradas |

A execução foi feita com Python 3.12.14, NumPy 2.1.3, PyTorch 2.5.1 CPU,
scikit-learn 1.5.2 e Matplotlib 3.9.2. Versões, hashes e contagens completas estão
em `eda_summary.json`.

### Forma de execução do notebook

O ambiente de validação bloqueia a abertura de sockets. Por isso, o comando
convencional `nbconvert --execute` não conseguiu iniciar o kernel, antes de
executar qualquer célula. A execução completa foi realizada por
`python scripts/run_eda.py`, usando um kernel IPython novo no próprio processo.
As saídas e imagens presentes no notebook foram produzidas pelas células reais.

Isso valida a execução sequencial do código com estado inicial limpo. A abertura
interativa no navegador/Jupyter convencional não foi testada neste ambiente.
O README documenta tanto o fluxo convencional quanto o executor sem sockets.

## Divisão obtida

Seed 42, uma classe anômala por cenário, 15% dos normais destinados à validação.

| Cenário | Treino | Validação | Teste normal | Teste anômalo | Excluídos do treino oficial |
|---|---:|---:|---:|---:|---:|
| Bag (8) | 45.900 | 8.100 | 9.000 | 1.000 | 6.000 Bag |
| Sandal (5) | 45.900 | 8.100 | 9.000 | 1.000 | 6.000 Sandal |

Em cada cenário, cada uma das nove classes normais tem 5.100 exemplos de treino
e 900 de validação. A divisão ocorre **após** remover a classe anômala.

A auditoria de sobreposição é feita por registro e arquivo de origem. Ela não
afirma que a base original esteja livre de imagens idênticas ou quase idênticas.

## Achados descritivos e hipóteses

As estatísticas abaixo usam somente os 60.000 exemplos do treino oficial.

- Sandal teve intensidade média de **0,1367** e **67,9%** de pixels exatamente
  iguais a zero; Sneaker teve média **0,1677** e **66,2%** de zeros. Isso sustenta
  investigar se o fundo reduz o erro médio mesmo quando a reconstrução do objeto
  é ruim. Não demonstra que o detector irá falhar.
- Bag teve média **0,3536** e **41,4%** de zeros. Sua intensidade média se sobrepõe
  à de roupas; não é correto tratá-la como facilmente separável apenas pelo brilho.
- As grades e imagens médias sugerem semelhanças entre T-shirt/top e Shirt,
  entre Pullover, Coat e Shirt, e entre as classes de calçados. As diferenças
  individuais continuam importantes, mesmo quando as médias se parecem.

Hipóteses pré-treino: Bag pode apresentar maior erro de reconstrução; Sandal pode
ter mais falsos negativos; imagens incomuns das classes normais podem gerar
falsos positivos. Os nomes “fácil” e “difícil” são expectativas a testar.

## O que explicar na apresentação

1. **Por que retirar uma classe?** Para simular uma categoria desconhecida durante
   o ajuste: o Autoencoder aprende a reconstruir apenas os padrões considerados normais.
2. **Por que manter validação sem anomalias?** Porque este é o protocolo one-class
   definido pelo grupo. Early stopping e limiar usam somente exemplos normais.
3. **Por que dividir por 255?** Para converter pixels uint8 de 0–255 para float32
   em 0–1, sem estimar parâmetros com o conjunto de teste.
4. **Por que seed e estratificação?** A seed repete a divisão no mesmo ambiente;
   a estratificação preserva a proporção das nove classes normais.
5. **Por que não confiar só na acurácia?** Responder “normal” para tudo acertaria
   90% do teste e não detectaria nenhuma anomalia.
6. **O que falta na etapa seguinte?** Treinar o modelo, escolher o limiar usando
   validação normal, congelar as decisões e avaliar no teste para verificar as hipóteses.

## Reproduzir

Com o ambiente virtual ativado e as dependências instaladas:

```bash
python scripts/run_eda.py
python -m pytest -q
python -m pip check
```

O primeiro comando baixa os arquivos ausentes, executa o notebook e atualiza as
figuras e o JSON. Os testes não acessam a internet; os testes de integração são
pulados se ainda não houver arquivos oficiais em `data/`.
