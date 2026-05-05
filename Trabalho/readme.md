# Análise e Predição de Preços de Imóveis em King County

## 1. Descrição do Projeto

Este projeto tem como objetivo realizar uma análise exploratória e preditiva sobre o dataset **House Sales in King County, USA**, disponível no Kaggle.

O conjunto de dados contém informações sobre imóveis vendidos em King County, região localizada no estado de Washington, nos Estados Unidos. A proposta principal é entender quais fatores mais influenciam o preço dos imóveis e desenvolver uma análise baseada em hipóteses, visualizações gráficas e modelos de Machine Learning para previsão de preços.

O projeto foi desenvolvido como um estudo de Ciência de Dados, passando pelas etapas de carregamento dos dados, exploração, limpeza, engenharia de atributos, análise de hipóteses, visualização e modelagem preditiva.

---

## 2. Objetivo

O objetivo principal é analisar os fatores que impactam o preço dos imóveis em King County.

Além disso, o projeto busca responder perguntas como:

- A localização influencia fortemente o preço dos imóveis?
- Casas maiores são sempre mais caras?
- Casas reformadas possuem maior valorização?
- Imóveis próximos à água são mais caros?
- A qualidade da construção tem impacto maior que a condição geral da casa?
- É possível prever o preço de um imóvel com base em suas características?

---

## 3. Dataset Utilizado

O dataset utilizado foi o **House Sales in King County, USA**.

Ele contém informações como:

- Preço do imóvel;
- Número de quartos;
- Número de banheiros;
- Área construída;
- Área do terreno;
- Número de andares;
- Vista para água;
- Qualidade da vista;
- Condição do imóvel;
- Qualidade da construção;
- Ano de construção;
- Ano de reforma;
- Código postal;
- Latitude e longitude;
- Informações dos imóveis vizinhos.

A variável alvo do projeto é:

```python
price
```

Ou seja, o objetivo da modelagem é prever o preço dos imóveis.

---

## 4. Principais Variáveis

| Variável | Descrição |
|---|---|
| `price` | Preço de venda do imóvel |
| `bedrooms` | Número de quartos |
| `bathrooms` | Número de banheiros |
| `sqft_living` | Área interna habitável |
| `sqft_lot` | Área total do terreno |
| `floors` | Número de andares |
| `waterfront` | Indica se o imóvel fica à beira d’água |
| `view` | Qualidade da vista |
| `condition` | Condição geral do imóvel |
| `grade` | Qualidade da construção e acabamento |
| `sqft_above` | Área construída acima do solo |
| `sqft_basement` | Área do porão |
| `yr_built` | Ano de construção |
| `yr_renovated` | Ano de reforma |
| `zipcode` | Código postal |
| `lat` | Latitude |
| `long` | Longitude |
| `sqft_living15` | Área média dos imóveis vizinhos |
| `sqft_lot15` | Área média dos terrenos vizinhos |

---

## 5. Hipóteses do Projeto

A análise foi guiada por 10 hipóteses principais.

### Hipótese 1: Imóveis com maior área habitável possuem preços mais altos

A variável `sqft_living` representa a área interna da casa. A hipótese é que imóveis maiores tendem a possuir preços mais elevados.

**Como analisar:**

- Gráfico de dispersão entre `sqft_living` e `price`;
- Cálculo de correlação;
- Comparação da importância da variável em modelos preditivos.

---

### Hipótese 2: A localização influencia fortemente o preço dos imóveis

As variáveis `lat`, `long` e `zipcode` indicam a localização do imóvel. A hipótese é que imóveis em regiões mais valorizadas possuem preços mais altos, independentemente do tamanho.

**Como analisar:**

- Mapa de dispersão usando latitude e longitude;
- Preço médio por `zipcode`;
- Visualização geográfica dos imóveis mais caros.

---

### Hipótese 3: Imóveis à beira d’água são mais caros

A variável `waterfront` indica se o imóvel está localizado próximo ou de frente para a água. A hipótese é que imóveis com essa característica possuem valorização significativa.

**Como analisar:**

- Comparar preço médio e mediano entre imóveis com e sem `waterfront`;
- Boxplot de `price` por `waterfront`;
- Analisar outliers de imóveis de luxo.

---

### Hipótese 4: A qualidade da construção influencia mais o preço do que a condição geral

A variável `grade` representa a qualidade da construção, enquanto `condition` representa o estado geral do imóvel. A hipótese é que `grade` tem impacto maior no preço do que `condition`.

**Como analisar:**

- Correlação de `grade` e `condition` com `price`;
- Gráficos de boxplot;
- Importância das variáveis em modelos de Machine Learning.

---

### Hipótese 5: Imóveis reformados possuem preços maiores

A variável `yr_renovated` indica se o imóvel passou por reforma. A hipótese é que imóveis reformados tendem a ser mais valorizados.

**Como analisar:**

- Criar a variável `was_renovated`;
- Comparar preço médio entre imóveis reformados e não reformados;
- Analisar o impacto da reforma considerando o ano de construção.

Exemplo de criação da variável:

```python
df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
```

---

### Hipótese 6: Imóveis mais novos tendem a ser mais caros

A variável `yr_built` informa o ano de construção. A hipótese é que imóveis mais novos possuem preço maior por apresentarem estrutura mais moderna.

**Como analisar:**

- Criar a variável `house_age`;
- Comparar idade do imóvel com preço;
- Analisar se imóveis antigos em boas localizações ainda mantêm preço elevado.

Exemplo:

```python
df["house_age"] = 2015 - df["yr_built"]
```

---

### Hipótese 7: O número de quartos não aumenta o preço de forma linear

A variável `bedrooms` representa o número de quartos. A hipótese é que, após determinado ponto, adicionar mais quartos não aumenta proporcionalmente o preço do imóvel.

**Como analisar:**

- Boxplot de `price` por número de quartos;
- Análise de preço médio por quantidade de quartos;
- Identificação de imóveis com muitos quartos e preços fora do padrão.

---

### Hipótese 8: A área dos imóveis vizinhos influencia o preço

A variável `sqft_living15` representa a área dos imóveis vizinhos mais próximos. A hipótese é que casas localizadas em regiões com imóveis maiores tendem a ser mais caras.

**Como analisar:**

- Correlação entre `sqft_living15` e `price`;
- Comparação entre `sqft_living` e `sqft_living15`;
- Análise do efeito de vizinhança.

---

### Hipótese 9: Imóveis com melhor vista possuem preços mais altos

A variável `view` representa a qualidade da vista do imóvel. A hipótese é que imóveis com melhor vista são mais valorizados.

**Como analisar:**

- Boxplot de `price` por nível de `view`;
- Comparar preço médio por categoria de vista;
- Verificar se a variável `view` está associada também a `waterfront`.

---

### Hipótese 10: O mês da venda pode influenciar o preço

A variável `date` contém a data da venda do imóvel. A hipótese é que o preço pode variar de acordo com o mês de venda, por causa de sazonalidade no mercado imobiliário.

**Como analisar:**

- Extrair o mês da venda;
- Comparar preço médio por mês;
- Verificar se existe tendência temporal.

Exemplo:

```python
df["date"] = pd.to_datetime(df["date"])
df["sale_month"] = df["date"].dt.month
```

---

## 6. Etapas do Projeto

O projeto foi dividido nas seguintes etapas:

### 6.1. Carregamento dos dados

Leitura do arquivo CSV e visualização inicial das informações.

```python
import pandas as pd

df = pd.read_csv("kc_house_data.csv")
df.head()
```

### 6.2. Análise exploratória dos dados

Nesta etapa foram analisadas:

- Dimensão do dataset;
- Tipos das variáveis;
- Dados ausentes;
- Estatísticas descritivas;
- Distribuição da variável `price`;
- Correlações entre variáveis.

### 6.3. Visualização dos dados

Foram gerados gráficos para entender melhor o comportamento dos dados, como:

- Histogramas;
- Boxplots;
- Gráficos de dispersão;
- Heatmap de correlação;
- Mapa usando latitude e longitude.

### 6.4. Engenharia de atributos

Foram criadas novas variáveis para melhorar a análise e a modelagem:

- `house_age`;
- `was_renovated`;
- `has_basement`;
- `living_lot_ratio`;
- `bath_per_bedroom`;
- `sale_month`.

### 6.5. Pré-processamento

Foram aplicadas técnicas como:

- Tratamento de dados ausentes;
- Padronização de variáveis numéricas;
- Codificação de variáveis categóricas;
- Separação entre treino e teste.

### 6.6. Modelagem preditiva

Foram testados modelos de regressão para prever o preço dos imóveis, como:

- Regressão Linear;
- Árvore de Decisão;
- Random Forest;
- Gradient Boosting.

### 6.7. Avaliação dos modelos

Os modelos foram avaliados com métricas como:

- MAE — Mean Absolute Error;
- RMSE — Root Mean Squared Error;
- R² — Coeficiente de determinação.

---

## 7. Visualização Geográfica

Como o dataset possui latitude e longitude, é possível visualizar os imóveis em um mapa de King County.

Exemplo básico:

```python
import matplotlib.pyplot as plt

df.plot(
    kind="scatter",
    x="long",
    y="lat",
    alpha=0.4,
    s=df["sqft_living"] / 100,
    c="price",
    cmap="viridis",
    colorbar=True,
    figsize=(10, 7)
)

plt.title("Distribuição dos imóveis em King County")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
```

Essa visualização ajuda a entender a relação entre localização e preço dos imóveis.

---

## 8. Exemplo de Pipeline de Machine Learning

```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

X = df.drop("price", axis=1)
y = df["price"]

cat_attribs = ["zipcode"]
num_attribs = [col for col in X.columns if col not in cat_attribs + ["date"]]

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

model = make_pipeline(
    preprocessing,
    RandomForestRegressor(random_state=42)
)
```

---

## 9. Tecnologias Utilizadas

- Python;
- Pandas;
- NumPy;
- Matplotlib;
- Seaborn;
- Scikit-learn;
- Jupyter Notebook;
- Kaggle Dataset.

---

## 10. Resultados Esperados

Ao final do projeto, espera-se:

- Identificar os principais fatores que influenciam o preço dos imóveis;
- Validar ou refutar as 10 hipóteses propostas;
- Construir visualizações claras para apresentação;
- Criar modelos de Machine Learning capazes de estimar preços de imóveis;
- Comparar o desempenho de diferentes modelos;
- Explicar os resultados de forma simples e objetiva.

---

## 11. Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd <nome-do-projeto>
```

### 2. Instale as dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Baixe o dataset

Baixe o arquivo `kc_house_data.csv` no Kaggle e coloque-o na pasta do projeto.

### 4. Execute o notebook

```bash
jupyter notebook
```

Abra o notebook principal e execute as células em sequência.

---

## 12. Estrutura Sugerida do Projeto

```text
projeto-king-county/
│
├── data/
│   └── kc_house_data.csv
│
├── notebooks/
│   └── analise_king_county.ipynb
│
├── images/
│   └── graficos_do_projeto.png
│
├── README.md
│
└── requirements.txt
```

---

## 13. Conclusão

Este projeto permite aplicar conceitos importantes de Ciência de Dados em um problema real de regressão: a previsão de preços de imóveis.

A partir das análises realizadas, é possível entender melhor como fatores como área construída, localização, qualidade da construção, vista, reforma e características da vizinhança influenciam o valor de uma casa.

Além da análise exploratória, o projeto também permite desenvolver e comparar modelos de Machine Learning, tornando-se uma boa base para estudos de regressão, engenharia de atributos e apresentação de hipóteses em Ciência de Dados.
