## R²

| Modelos | Média | Desvio-Padrão | Maior Valor | Menor Valor |
| --- | --- | --- | --- | --- |
| Polinomial | 0.9862 | 0.0204 | 0.9997 | 0.8135 |
| MQO tradicional | -0.4159 | 2.5811 | 0.7462 | -20.2422 |
| MQO regularizado (0.25) | -0.8423 | 5.2781 | 0.7185 | -84.2001 |
| MQO regularizado (0.50) | -1.1172 | 6.5316 | 0.7704 | -88.6088 |
| MQO regularizado (0.75) | -1.2322 | 9.0272 | 0.7883 | -124.9546 |
| MQO regularizado (1.00) | -0.6804 | 5.1989 | 0.7520 | -68.8973 |

## MSE

| Modelos | Média | Desvio-Padrão | Maior Valor | Menor Valor |
| --- | --- | --- | --- | --- |
| Polinomial | 0.0078 | 0.0074 | 0.0452 | 0.0003 |
| MQO tradicional | 0.5275 | 0.3092 | 1.9655 | 0.1517 |
| MQO regularizado (0.25) | 0.5524 | 0.3121 | 1.9318 | 0.1481 |
| MQO regularizado (0.50) | 0.5291 | 0.3278 | 2.3324 | 0.1136 |
| MQO regularizado (0.75) | 0.5178 | 0.3367 | 2.1117 | 0.1244 |
| MQO regularizado (1.00) | 0.5188 | 0.3205 | 2.3573 | 0.1012 |

```
[Regressao Polinomial (q=5)] MSE - Media: 0.0078, Desvio-Padrao: 0.0074, Maior: 0.0452, Menor: 0.0003
[Regressao Polinomial (q=5)] R2 - Media: 0.9862, Desvio-Padrao: 0.0204, Maior: 0.9997, Menor: 0.8135
[Regressao Linear (MQO Tradicional)] MSE - Media: 0.5275, Desvio-Padrao: 0.3092, Maior: 1.9655, Menor: 0.1517
[Regressao Linear (MQO Tradicional)] R2 - Media: -0.4159, Desvio-Padrao: 2.5811, Maior: 0.7462, Menor: -20.2422
[Regressao Ridge (lambda=0.25)] MSE - Media: 0.5524, Desvio-Padrao: 0.3121, Maior: 1.9318, Menor: 0.1481
[Regressao Ridge (lambda=0.25)] R2 - Media: -0.8423, Desvio-Padrao: 5.2781, Maior: 0.7185, Menor: -84.2001
[Regressao Ridge (lambda=0.5)] MSE - Media: 0.5291, Desvio-Padrao: 0.3278, Maior: 2.3324, Menor: 0.1136
[Regressao Ridge (lambda=0.5)] R2 - Media: -1.1172, Desvio-Padrao: 6.5316, Maior: 0.7704, Menor: -88.6088
[Regressao Ridge (lambda=0.75)] MSE - Media: 0.5178, Desvio-Padrao: 0.3367, Maior: 2.1117, Menor: 0.1244
[Regressao Ridge (lambda=0.75)] R2 - Media: -1.2322, Desvio-Padrao: 9.0272, Maior: 0.7883, Menor: -124.9546
[Regressao Ridge (lambda=1)] MSE - Media: 0.5188, Desvio-Padrao: 0.3205, Maior: 2.3573, Menor: 0.1012
[Regressao Ridge (lambda=1)] R2 - Media: -0.6804, Desvio-Padrao: 5.1989, Maior: 0.7520, Menor: -68.8973
```

## Classificação (Validação por Monte Carlo, R = 500)

| Modelos                        | Média  | Desvio-Padrão | Maior Valor | Menor Valor |
|---------------------------------|--------|----------------|-------------|-------------|
| MQO tradicional                 | 0.7239 | 0.0064         | 0.7418      | 0.7013      |
| MQO regularizado (λ = 1,00)     | 0.7239 | 0.0064         | 0.7418      | 0.7013      |
| MQO polinomial (q = 4)          | 0.9969 | 0.0005         | 0.9983      | 0.9955      |

Os resultados indicam que as classes do problema não são linearmente separáveis no espaço original dos dois sensores: o MQO tradicional e o MQO regularizado, que são modelos lineares, ficam limitados a uma acurácia média de apenas 72,39%, enquanto o MQO polinomial (q = 4) atinge 99,69%, mostrando que fronteiras de decisão curvas são necessárias para separar bem as 5 classes de expressão facial. O MQO regularizado apresentou resultado praticamente idêntico ao MQO tradicional porque, para λ = 1, o termo de penalização é desprezível frente à magnitude de XᵀX (da ordem de 10¹⁰–10¹¹, dado que os sinais dos sensores chegam a ~4000 e há 50.000 amostras); nesse regime, a regularização de Tikhonov não altera de forma perceptível o vetor de parâmetros estimado. Já o desvio-padrão do modelo polinomial (0,0005) é uma ordem de grandeza menor que o dos modelos lineares (0,0064), o que mostra que, além de mais acurado, ele é também mais estável entre as 500 rodadas de validação por Monte Carlo — um indicativo de que o grau q = 4, escolhido pelo critério de poda do item 4, oferece um bom compromisso entre complexidade do modelo e capacidade de generalização, sem sinais de overfitting nesse conjunto de teste.