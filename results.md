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

## Regressão — avaliação de q (item 4)

Saída do bloco de avaliação de q em `src/regression_main.py`: as mesmas 500 partições 80/20 são usadas para todos os valores de q, sorteadas com `np.random.default_rng(42)` (reprodutível). As partições da validação principal (tabelas acima) não usam semente, por isso o q = 5 daqui difere um pouco da linha "Polinomial".

| q | R² médio | R² mínimo | MSE médio | MSE máximo |
| --- | --- | --- | --- | --- |
| 1 | -1.3214 | -167.0788 | 0.5505 | 2.7636 |
| 2 | -0.0213 | -69.2665 | 0.2008 | 1.4026 |
| 3 | 0.7304 | -11.1990 | 0.0484 | 0.5066 |
| 4 | 0.9479 | -2.0416 | 0.0093 | 0.1133 |
| 5 | 0.9850 | 0.8207 | 0.0078 | 0.0562 |
| 6 | 0.9760 | 0.4038 | 0.0118 | 0.2713 |
| 7 | 0.9696 | -0.3053 | 0.0123 | 0.7295 |
| 8 | 0.9786 | -0.6265 | 0.0069 | 0.4566 |
| 9 | 0.9949 | 0.8159 | 0.0024 | 0.2560 |
| 10 | 0.9865 | -2.4014 | 0.0213 | 9.0452 |

```
[Selecao de q] q=1: R2 medio=-1.3214, R2 minimo=-167.0788, MSE medio=0.5505, MSE maximo=2.7636
[Selecao de q] q=2: R2 medio=-0.0213, R2 minimo=-69.2665, MSE medio=0.2008, MSE maximo=1.4026
[Selecao de q] q=3: R2 medio=0.7304, R2 minimo=-11.1990, MSE medio=0.0484, MSE maximo=0.5066
[Selecao de q] q=4: R2 medio=0.9479, R2 minimo=-2.0416, MSE medio=0.0093, MSE maximo=0.1133
[Selecao de q] q=5: R2 medio=0.9850, R2 minimo=0.8207, MSE medio=0.0078, MSE maximo=0.0562
[Selecao de q] q=6: R2 medio=0.9760, R2 minimo=0.4038, MSE medio=0.0118, MSE maximo=0.2713
[Selecao de q] q=7: R2 medio=0.9696, R2 minimo=-0.3053, MSE medio=0.0123, MSE maximo=0.7295
[Selecao de q] q=8: R2 medio=0.9786, R2 minimo=-0.6265, MSE medio=0.0069, MSE maximo=0.4566
[Selecao de q] q=9: R2 medio=0.9949, R2 minimo=0.8159, MSE medio=0.0024, MSE maximo=0.2560
[Selecao de q] q=10: R2 medio=0.9865, R2 minimo=-2.4014, MSE medio=0.0213, MSE maximo=9.0452
```

## Classificação — seleção de q (item 4)

Saída de `select_polynomial_degree` em `src/classification_main.py` (partição única 80/20, `random_state=42`, tolerância de 0,002). As acurácias são reprodutíveis; os tempos vêm de uma única execução e variam com o equipamento.

| q | coeficientes por classe | acurácia | tempo (s) |
| --- | --- | --- | --- |
| 1 | 3 | 0.7253 | 0.0042 |
| 2 | 6 | 0.9522 | 0.0070 |
| 3 | 10 | 0.9857 | 0.0140 |
| 4 | 15 | 0.9965 | 0.0223 |
| 5 | 21 | 0.7976 | 0.0334 |
| 6 | 28 | 0.7900 | 0.0512 |

q escolhido: 4. Na mesma execução, a validação de Monte Carlo reproduziu exatamente a tabela abaixo.

## Classificação (Validação por Monte Carlo, R = 500)

| Modelos                        | Média  | Desvio-Padrão | Maior Valor | Menor Valor |
|---------------------------------|--------|----------------|-------------|-------------|
| MQO tradicional                 | 0.7239 | 0.0064         | 0.7418      | 0.7013      |
| MQO regularizado (λ = 1,00)     | 0.7239 | 0.0064         | 0.7418      | 0.7013      |
| MQO polinomial (q = 4)          | 0.9969 | 0.0005         | 0.9983      | 0.9955      |

Os resultados indicam que as classes do problema não são linearmente separáveis no espaço original dos dois sensores: o MQO tradicional e o MQO regularizado, que são modelos lineares, ficam limitados a uma acurácia média de apenas 72,39%, enquanto o MQO polinomial (q = 4) atinge 99,69%, mostrando que fronteiras de decisão curvas são necessárias para separar bem as 5 classes de expressão facial. O MQO regularizado apresentou resultado praticamente idêntico ao MQO tradicional porque, para λ = 1, o termo de penalização é desprezível frente à magnitude de XᵀX (da ordem de 10¹⁰–10¹¹, dado que os sinais dos sensores chegam a ~4000 e há 50.000 amostras); nesse regime, a regularização de Tikhonov não altera de forma perceptível o vetor de parâmetros estimado. Já o desvio-padrão do modelo polinomial (0,0005) é uma ordem de grandeza menor que o dos modelos lineares (0,0064), o que mostra que, além de mais acurado, ele é também mais estável entre as 500 rodadas de validação por Monte Carlo — um indicativo de que o grau q = 4, escolhido pelo critério de poda do item 4, oferece um bom compromisso entre complexidade do modelo e capacidade de generalização, sem sinais de overfitting nesse conjunto de teste.