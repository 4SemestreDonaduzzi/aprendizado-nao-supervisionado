# Relatório — Projeto Final 1: Autoencoder

> Compressão, reconstrução e detecção de anomalias no Fashion-MNIST.
>
> **Status:** esqueleto. Seções marcadas com ⏳ dependem de Issues ainda abertas.
> Regra do documento: **todo número citado aqui vem de um arquivo em `reports/`**,
> gerado por um experimento que de fato rodou. Nada estimado ou arredondado à mão.

Para instalar e executar o projeto, veja o [README](../README.md).

---

## 1. O problema

<!--
Responder em 1–2 parágrafos:
- O que é detecção de anomalias e por que ela é um problema de aprendizado NÃO supervisionado?
- Por que, no mundo real, é comum ter muitos exemplos "normais" e poucos (ou nenhum) exemplos de anomalia?
- O que o professor pediu (compressão, reconstrução, espaço latente, erro, anomalias)?
-->

## 2. Por que o Fashion-MNIST

<!--
- O que é a base: 70 mil imagens 28×28 em tons de cinza, 10 classes de roupas e acessórios.
- Por que ela e não outra (visualização intuitiva, rótulos para medir a detecção, treina em CPU).
- A LIMITAÇÃO, dita com honestidade aqui mesmo: a anomalia é artificial (uma classe
  que o modelo nunca viu), não um defeito real. O que isso permite e o que NÃO permite concluir?
-->

## 3. Como funciona um Autoencoder

<!--
Esta é a seção que o professor mais vai perguntar. Escrever com as próprias palavras.
Responder, nesta ordem:
1. O que são encoder, espaço latente (gargalo) e decoder?
2. Por que um modelo que só aprende a copiar a entrada NÃO aprende simplesmente a identidade?
3. O que é o erro de reconstrução e por que ele vira um "score" de anomalia?
4. Por que treinar só com dados normais faz o modelo reconstruir mal o que é diferente?
5. Quando essa ideia FALHA? (anomalia que o modelo reconstrói bem)
Sugestão: um diagrama simples (entrada 784 → ... → latente → ... → saída 784).
-->

## 4. Metodologia

### 4.1 Dados e divisão dos conjuntos

<!--
Fonte: Issue #2, README (seção Dados) e reports/eda_summary.json.
- Estratégia one-class: a classe anômala é removida ANTES de dividir treino/validação.
- Tabela dos conjuntos (copiar os números de eda_summary.json, não de memória).
- Dois cenários: Bag (8) e Sandal (5). Por que dois?
-->

### 4.2 Como evitamos vazamento de dados

<!--
- O que é vazamento de dados e por que ele gera resultados "bonitos e falsos".
- As três regras do projeto:
  (a) treino e validação nunca contêm a classe anômala;
  (b) o limiar é escolhido SÓ com erros da validação;
  (c) o teste é usado apenas uma vez, para medir.
- Como verificamos (testes automatizados de #2 e #6).
-->

### 4.3 Análise exploratória e hipóteses (registradas antes do treino)

<!--
Fonte: Issue #3, notebooks/01_eda.ipynb, reports/validacao_dados_eda.md.
- Resumir as hipóteses H1, H2... COM A DATA em que foram registradas (PR #12, 08/11/2026).
- Por que registrar antes do treino importa? (evita ajustar a hipótese ao resultado)
- Linha de base nos pixels brutos (PR #13): PCA e t-SNE antes de qualquer modelo.
-->

### 4.4 Arquitetura do Autoencoder ⏳ #4

<!--
- Camadas, dimensão latente, ativações, nº de parâmetros.
- Função de perda escolhida (MSE ou BCE) e por quê.
-->

### 4.5 Treinamento ⏳ #5

<!--
- Otimizador, learning rate, batch size, épocas, early stopping.
- Figura: curvas de loss de treino e validação. Houve overfitting?
- Tempo de treino e hardware.
-->

### 4.6 Limiar e métricas ⏳ #6

<!--
- Como o limiar foi definido (percentil da validação) e por que esse percentil.
- O que cada métrica mede e por que usamos: MSE/MAE, ROC-AUC, PR-AUC, Precision/Recall/F1.
- Linha de base: detector por erro de reconstrução do PCA (para comparar com o Autoencoder).
-->

## 5. Resultados ⏳ #6, #7, #8, #9

### 5.1 Reconstrução

<!-- Figura original | reconstruído | mapa de erro. Métricas por classe. -->

### 5.2 Detecção de anomalias

<!-- Tabela: cenário Bag vs. Sandal, Autoencoder vs. PCA. Histograma dos erros. -->

### 5.3 Espaço latente

<!-- PCA e t-SNE do espaço latente vs. dos pixels brutos. Métrica de vizinhos (não só a figura). -->

### 5.4 Exemplos de falhas

<!-- Falsos positivos e falsos negativos, cada um com o provável motivo. -->

### 5.5 Denso vs. convolucional (opcional) ⏳ #9

## 6. Análise crítica ⏳

<!--
Escrever por último, mas pensar desde já:
- As hipóteses da seção 4.3 se confirmaram? Quais não, e por quê?
- Quando o Autoencoder funcionou e quando falhou?
- O Autoencoder foi melhor que a linha de base do PCA? Se não, o que isso diz?
- A anomalia artificial (classe nunca vista): até onde os resultados valem para um problema real?
- O que faríamos diferente com mais tempo?
-->

## 7. Limitações

<!-- Lista honesta: dataset, anomalia artificial, CPU, uma seed, etc. -->

## 8. Roteiro da apresentação

<!--
1. Problema (1 slide)
2. Como funciona um Autoencoder (1–2 slides)
3. Dados e estratégia contra vazamento (1 slide)
4. Arquitetura e treinamento (1 slide)
5. Resultados: reconstrução, detecção, espaço latente (2–3 slides)
6. Demonstração ao vivo (Streamlit, #10)
7. Falhas e análise crítica (1–2 slides)
8. Conclusão
Quem apresenta cada parte?
-->
