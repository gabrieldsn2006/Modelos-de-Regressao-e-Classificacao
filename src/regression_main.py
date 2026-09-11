import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('src/dados/china_gdp.csv', delimiter=',', skiprows=1)

N = data.shape[0]       # número de observações
p = data.shape[1] - 1   # número de variáveis independentes 

X = data[:, 0].reshape(N, p)  # Ano (variável independente)
Y = data[:, -1].reshape(N, 1) # PIB (variável dependente)

# scatter plot dos dados
plt.figure(1)
plt.scatter(X, Y, color='b', label='Dados Reais')
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

# Padronização (Z-score)
media_x = np.mean(X)
desv_pad_x = np.std(X)
X = (X-media_x)/desv_pad_x
Y = (Y-np.mean(Y))/np.std(Y)


def plot_model(X, Y, model, title, i):
    plt.figure(i)
    plt.scatter(X, Y, color='b', label='Dados Reais')
    plt.xlabel('Ano'); plt.ylabel('PIB (em bilhões de dólares)')
    plt.title(title); plt.legend()
    plt.grid()
    
    x = np.linspace(-2.5, 2.5, 200)
    y_pred = model.predict(x.reshape(len(x),1))
    plt.plot(x, y_pred, c='orange', lw=3, label='Modelo Ajustado')
    plt.legend()

def print_metrics(metrics, model_name):
    mse_values = [m[0] for m in metrics]
    r2_values  = [m[1] for m in metrics]
    
    print(f"[{model_name}] MSE - Media: {np.mean(mse_values):.4f}, Desvio-Padrao: {np.std(mse_values):.4f}, Maior: {np.max(mse_values):.4f}, Menor: {np.min(mse_values):.4f}")
    print(f"[{model_name}] R2 - Media: {np.mean(r2_values):.4f}, Desvio-Padrao: {np.std(r2_values):.4f}, Maior: {np.max(r2_values):.4f}, Menor: {np.min(r2_values):.4f}")

# avaliação inicial do hiperparâmetro q (estratégia de poda comparando o R²)
# as mesmas 500 partições (semente fixa) são usadas para todos os valores de q
rng_q = np.random.default_rng(42)
particoes = [rng_q.permutation(N) for _ in range(500)]
for q_teste in range(1, 11):
    q_metrics = []
    for idx in particoes:
        Xr = np.copy(X)[idx,:]
        Yr = np.copy(Y)[idx,:]

        X_treino = Xr[:int(.8*N),:]
        Y_treino = Yr[:int(.8*N),:]

        X_teste = Xr[int(.8*N):,:]
        Y_teste = Yr[int(.8*N):,:]

        pr = PolynomialRegression(X_treino, Y_treino, q=q_teste)
        pr.fit()

        Y_pred = pr.predict(X_teste)
        E = Y_teste - Y_pred
        SSE = np.sum(E**2)
        MSE = np.mean(E**2)
        y_bar = np.mean(Y_teste)
        SST = np.sum((Y_teste - y_bar)**2)
        R2 = 1 - SSE/SST
        q_metrics.append((MSE, R2))
    mse_q = [m[0] for m in q_metrics]
    r2_q  = [m[1] for m in q_metrics]
    print(f"[Selecao de q] q={q_teste}: R2 medio={np.mean(r2_q):.4f}, R2 minimo={np.min(r2_q):.4f}, MSE medio={np.mean(mse_q):.4f}, MSE maximo={np.max(mse_q):.4f}")

# polynomial regression
q = 5
pr_metrics = []
for i in range(500):
    idx = np.random.permutation(N)
    Xr = np.copy(X)[idx,:]
    Yr = np.copy(Y)[idx,:]
    
    X_treino = Xr[:int(.8*N),:]
    Y_treino = Yr[:int(.8*N),:]
    
    X_teste = Xr[int(.8*N):,:]
    Y_teste = Yr[int(.8*N):,:]
    
    pr = PolynomialRegression(X_treino, Y_treino, q=q)
    pr.fit()
    
    Y_pred = pr.predict(X_teste)
    E = Y_teste - Y_pred
    SSE = np.sum(E**2)
    MSE = np.mean(E**2)
    y_bar = np.mean(Y_teste)
    SST = np.sum((Y_teste - y_bar)**2)
    R2 = 1 - SSE/SST
    pr_metrics.append((MSE, R2))
plot_model(X, Y, pr, f"Regressao Polinomial (q={q})", 2)
print_metrics(pr_metrics, f"Regressao Polinomial (q={q})")

# linear regression
lr_1_metrics = []
for i in range(500):
    idx = np.random.permutation(N)
    Xr = np.copy(X)[idx,:]
    Yr = np.copy(Y)[idx,:]
    
    X_treino = Xr[:int(.8*N),:]
    Y_treino = Yr[:int(.8*N),:]
    
    X_teste = Xr[int(.8*N):,:]
    Y_teste = Yr[int(.8*N):,:]
    
    lr = LinearRegression(X_treino, Y_treino, fit_intercept=True, solver='OLS')
    lr.fit()
    
    Y_pred = lr.predict(X_teste)
    E = Y_teste - Y_pred
    SSE = np.sum(E**2)
    MSE = np.mean(E**2)
    y_bar = np.mean(Y_teste)
    SST = np.sum((Y_teste - y_bar)**2)
    R2 = 1 - SSE/SST
    lr_1_metrics.append((MSE, R2))
plot_model(X, Y, lr, "Regressao Linear (MQO Tradicional)", 3)
print_metrics(lr_1_metrics, "Regressao Linear (MQO Tradicional)")

# ridge regression
ridge_metrics = {0.25: [], 0.5: [], 0.75: [], 1: []}
for i, lambd in enumerate(ridge_metrics.keys()):
    for _ in range(500):
        idx = np.random.permutation(N)
        Xr = np.copy(X)[idx,:]
        Yr = np.copy(Y)[idx,:]
        
        X_treino = Xr[:int(.8*N),:]
        Y_treino = Yr[:int(.8*N),:]
        
        X_teste = Xr[int(.8*N):,:]
        Y_teste = Yr[int(.8*N):,:]
        
        ridge = LinearRegression(X_treino, Y_treino, fit_intercept=True, solver='Ridge', lambd=lambd)
        ridge.fit()
        
        Y_pred = ridge.predict(X_teste)
        E = Y_teste - Y_pred
        SSE = np.sum(E**2)
        MSE = np.mean(E**2)
        y_bar = np.mean(Y_teste)
        SST = np.sum((Y_teste - y_bar)**2)
        R2 = 1 - SSE/SST
        ridge_metrics[lambd].append((MSE, R2))
    plot_model(X, Y, ridge, f"Regressao Ridge (lambda={lambd})", 4 + i)
    print_metrics(ridge_metrics[lambd], f"Regressao Ridge (lambda={lambd})")


plt.show()
bp = 1