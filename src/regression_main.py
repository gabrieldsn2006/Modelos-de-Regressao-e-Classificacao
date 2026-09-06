import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('src/dados/china_gdp.csv', delimiter=',', skiprows=1)

N = data.shape[0]       # número de observações
p = data.shape[1] - 1   # número de variáveis independentes 

x = data[:, 0].reshape(N, p)  # Ano (variável independente)
y = data[:, -1].reshape(N, 1) # PIB (variável dependente)

# scatter plot dos dados
plt.scatter(x, y, color='b', label='Dados Reais')
plt.xlabel('Ano'); plt.ylabel('PIB (em bilhões de dólares)')
plt.title('Relação entre Ano e PIB da China'); plt.legend()
plt.grid()

# Modelos a serem implementados (todos estimam o valor do intercepto)
# 1. MQO tradicional
# 2. MQO com regularização (Tikhonov, com hiperparâmetro λ, valores para testar: {0, .25, .5, .75, 1})
# 3. Regressão Polinomial via MQO (com hiperparâmetro 'q')

# Utilizar Random Subsampling Validation por R=500 rodadas (80/20)
# Métricas: MSE e R²

# Montar a tabela de avaliação dos modelos
# Modelos       | Média | Desvio-Padrão | Maior | Menor
# Poly          | ...
# MQO trad      | ...
# MQO reg .25   | ...
# MQO reg .5    | ...
# MQO reg .75   | ...
# MQO reg 1     | ...

# Discutir os resultados e usar outros gráficos se for interessante

from AV1_kitlearn import *
# TO DO



plt.show()

bp = 1