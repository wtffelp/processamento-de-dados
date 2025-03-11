
"""
**Questão: Previsão de Vendas em uma Lanchonete**

Você recebeu um conjunto de dados de uma rede de lanchonetes contendo informações diárias dos últimos 3 meses. A base de dados possui as seguintes colunas:

- **Data:** Data da coleta dos dados.
- **Dia da Semana:** Dia da semana (Segunda, Terça, etc.).
- **Temperatura:** Temperatura média do dia (°C).
- **Feriado:** Indicação se o dia foi feriado (Sim/Não).
- **Vendas:** Número total de vendas realizadas no dia.

A seguir, um exemplo ilustrativo (tabela simplificada):

| Data       | Dia da Semana | Temperatura (°C) | Feriado | Vendas |
|------------|---------------|------------------|---------|--------|
| 2025-01-01 | Quarta        | 28               | Sim     | 150    |
| 2025-01-02 | Quinta        | 30               | Não     | 200    |
| 2025-01-03 | Sexta         | 27               | Não     | 250    |
| 2025-01-04 | Sábado        | 25               | Não     | 300    |
| 2025-01-05 | Domingo       | 26               | Não     | 280    |

**Tarefa:**

1. **Análise Exploratória dos Dados (EDA):**  
   - Investigue o conjunto de dados para identificar tendências, padrões e anomalias.  
   - Verifique se há dados faltantes ou inconsistências e proponha estratégias de limpeza e tratamento.

2. **Pré-processamento dos Dados:**  
   - Execute as transformações necessárias, como conversão de variáveis (ex.: transformar "Feriado" em variável numérica) e normalização, se aplicável.
   - Considere a criação de novas variáveis que possam ajudar na análise (ex.: categorização dos dias da semana).

3. **Modelagem Preditiva:**  
   - Utilize técnicas de machine learning para construir um modelo capaz de prever o número de vendas para os próximos 7 dias.  
   - Justifique a escolha do modelo (por exemplo, regressão linear, árvores de decisão, etc.) e apresente as métricas de avaliação (ex.: RMSE, MAE).

4. **Documentação:**  
   - Elabore um relatório em formato de texto descrevendo sua metodologia. Inclua:  
     - **Passo a passo da análise:** desde o entendimento do problema, pré-processamento, análise exploratória até a modelagem e previsão.  
     - **Ferramentas utilizadas:** indique se usou Python (com Scikit-Learn, Pandas, Matplotlib, etc.), Orange, ou outra ferramenta.  
     - **Justificativas das escolhas:** explique por que optou por determinados métodos e como os resultados foram alcançados.

---

**Instruções Básicas:**

- **Leitura e Compreensão:**  
  Comece entendendo o contexto e as variáveis do conjunto de dados. Observe as relações entre temperatura, feriados e vendas.

- **Preparação dos Dados:**  
  Trate os dados faltantes, converta variáveis categóricas em numéricas (se necessário) e normaliza os dados para melhorar a performance do modelo.

- **Análise Exploratória:**  
  Utilize gráficos (barras, linhas, histogramas) para visualizar tendências e padrões. Essa etapa pode ajudar a identificar quais variáveis possuem maior influência nas vendas.

- **Construção e Validação do Modelo:**  
  Divida seu conjunto de dados em treinamento e teste. Escolha um ou mais modelos de machine learning, ajuste seus parâmetros e avalie a performance com métricas adequadas.

- **Documentação do Processo:**  
  No relatório final, descreva cada etapa com clareza. Explique a lógica por trás das transformações, a escolha dos modelos e como as métricas foram calculadas. Esse relatório é tão importante quanto os resultados numéricos.

- **Entrega:**  
  Apresente tanto o código/ferramenta utilizada quanto o relatório descritivo, garantindo que um avaliador consiga seguir seu raciocínio e reproduzir os resultados.

---

Esta questão é projetada para avaliar sua capacidade de interpretar dados reais, aplicar técnicas de análise e machine learning, e, principalmente, documentar todo o processo de forma clara e detalhada. Boa sorte!
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


# Criando um dataset de exemplo (dados dos últimos 5 dias)
data = {
    'Data': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05', '2025-01-06'],
    'Dia da Semana': ['Terça','Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'],
    'Temperatura (°C)': [31 ,28, 30, 27, 25, 26],
    'Feriado': ['Sim', 'Sim', 'Não', 'Não', 'Não', 'Não'],
    'Vendas': [270, 150, 200, 250, 300, 280]
}
df = pd.DataFrame(data)
df['Data'] = pd.to_datetime(df['Data'])

# exibindo as primeiras linhas
print("Dataset original:")
print(df)

# convertendo 'Feriado': Sim = 1, Não = 0
df['Feriado_bin'] = df['Feriado'].apply(lambda x: 1 if x == 'Sim' else 0)

# convertendo 'Dia da Semana' para variáveis dummy (drop_first para evitar multicolinearidade)
df_dummies = pd.get_dummies(df['Dia da Semana'], prefix='Dia', drop_first=True)
df = pd.concat([df, df_dummies], axis=1)

print("\nDataset após pré-processamento:")
print(df)


# selecionando as features e a variável alvo
feature_cols = ['Temperatura (°C)', 'Feriado_bin'] + list(df_dummies.columns)
X = df[feature_cols]
y = df['Vendas']

# dividindo os dados em conjunto de treinamento e teste (mesmo que com poucos exemplos)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# criando e treinando o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# calculando o erro médio quadrático (MSE)
mse = mean_squared_error(y_test, y_pred)
print("\nMSE no conjunto de teste:", mse)

# criando um dataset para os próximos 7 dias
future_data = {
    'Data': pd.date_range(start=df['Data'].max() + pd.Timedelta(days=1), periods=7),
    'Dia da Semana': ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'],
    'Temperatura (°C)': [26, 27, 28, 29, 30, 27, 26],
    'Feriado': ['Não'] * 7  # Supondo que não há feriado nesses dias
}
df_future = pd.DataFrame(future_data)
df_future['Feriado_bin'] = df_future['Feriado'].apply(lambda x: 1 if x == 'Sim' else 0)

# criando variáveis dummy para 'Dia da Semana'
df_future_dummies = pd.get_dummies(df_future['Dia da Semana'], prefix='Dia', drop_first=True)
df_future = pd.concat([df_future, df_future_dummies], axis=1)

# garantindo que o df_future tenha as mesmas colunas dummy que o dataset original
dummy_cols = df_dummies.columns.tolist()
for col in dummy_cols:
    if col not in df_future.columns:
        df_future[col] = 0  # Preencher com 0 se a coluna não existir

# organizando as colunas na msm ordem q usou no modelo
df_future = df_future[['Temperatura (°C)', 'Feriado_bin'] + dummy_cols]

# previsao de vendas dos proximos 7 dias
future_predictions = model.predict(df_future)
df_future['Vendas_Previstas'] = future_predictions

print("\nPrevisão para os próximos 7 dias:")
# exibindo a previsão junto com os dados de entrada
df_future_display = df_future.copy()
df_future_display.index = pd.date_range(start=df['Data'].max() + pd.Timedelta(days=1), periods=7)
print(df_future_display)

# Supondo que y_test e y_pred já estejam definidos
# y_test = [...]
# y_pred = [...]

# Cálculo do erro médio quadrático (MSE)
mse = mean_squared_error(y_test, y_pred)
print("\nMSE no conjunto de teste:", mse)

# Calcular os limites dos eixos
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))

# Criar o gráfico de dispersão
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', label='Previsões')

# Plotar a linha de identidade
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal')

# Configurar os limites dos eixos
plt.xlim([min_val - 10, max_val + 10])
plt.ylim([min_val - 10, max_val + 10])

# Adicionar rótulos e título
plt.xlabel("Vendas Reais")
plt.ylabel("Vendas Preditas")
plt.title("Comparação: Vendas Reais vs. Preditas")
plt.legend()
plt.grid(True)
plt.show()

