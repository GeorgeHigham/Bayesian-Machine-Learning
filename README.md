# Bayesian-Machine-Learning
Using Bayesian Machine Learning as a predictor for energy efficiency

## Data

| Feature                      | Variable Name |
|------------------------------|---------------|
| Constant                     | X1            |
| Relative Compactness         | X2            |
| Surface Area                 | X3            |
| Wall Area                    | X4            |
| Roof Area                    | X5            |
| Overall Height               | X6            |
| Orientation                  | X7            |
| Glazing Area                 | X8            |
| Glazing Area Distribution    | X9            |

| Target       | Variable Name |
|--------------|---------------|
| Heating Load | Y1            |
| Cooling Load | Y2            |


## Results

### Heating
Bayesian Linear Regression                      
| Metric          | Training             | Test                |
|-----------------|----------------------|---------------------|
| RMSE            | 2.8307664611009185   | 4.0561986640826015  |
| MAE             | 2.019417727018286    | 3.4091833723517095  |

HMC Sampling for Linear Regression               
| Metric          | Training             | Test                 |
|-----------------|----------------------|----------------------|
| RMSE            | 2.846183953103809    | 3.971034977449392    |
| MAE             | 2.0254611262483677   | 3.3257646020642615   |

HMC for Classification                
| Metric                    | Training       | Test  |
|---------------------------|----------------|-------|
| Classification Error      | 2.34375%       | 0.0%  |

Variational Inference                           
| Metric          | Training             | Test                |
|-----------------|----------------------|---------------------|
| RMSE            | 2.830790843455396    | 4.055688354233558   |
| MAE             | 2.0194079559741316   | 3.4085659291551345  |

### Cooling
Bayesian Linear Regression                      
| Metric          | Training             | Test                |
|-----------------|----------------------|---------------------|
| RMSE            | 3.131454032276755    | 3.600741272794866   |
| MAE             | 2.1810560236260756   | 2.8074050566618816  |

HMC Sampling for Linear Regression               
| Metric          | Training             | Test                 |
|-----------------|----------------------|----------------------|
| RMSE            | 3.147003758619202    | 3.5493992005477257   |
| MAE             | 2.194362611395043    | 2.6963719848256176   |

HMC for Classification                     
| Metric                    | Training       | Test       |
|---------------------------|----------------|------------|
| Classification Error      | 1.302083333%   | 0.0%       |

Variational Inference                           
| Metric          | Training             | Test                |
|-----------------|----------------------|---------------------|
| RMSE            | 3.1314217888941567   | 3.6012172390656616  |
| MAE             | 2.1810952861256894   | 2.808840257231868   |

## References 

Data: Tsanas,Athanasios and Xifara,Angeliki. (2012). Energy Efficiency. UCI Machine Learning Repository. https://doi.org/10.24432/C51307.

Sampling code from from University of Bath, Bayesian Machine Learning CM50268 based on Radford Neal "one_step" code: http://www.cs.utoronto.ca/~radford/


