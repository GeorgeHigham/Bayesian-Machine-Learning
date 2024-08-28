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
| Metric          | Training   | Test      |
|-----------------|------------|-----------|
| RMSE            | 2.83077    | 4.05620   |
| MAE             | 2.01942    | 3.40918   |

HMC Sampling for Linear Regression               
| Metric          | Training   | Test      |
|-----------------|------------|-----------|
| RMSE            | 2.84618    | 3.97103   |
| MAE             | 2.02546    | 3.32576   |

HMC for Classification                
| Metric                    | Training | Test |
|---------------------------|----------|------|
| Classification Error      | 2.34375% | 0.0% |

Variational Inference                           
| Metric          | Training   | Test      |
|-----------------|------------|-----------|
| RMSE            | 2.83079    | 4.05569   |
| MAE             | 2.01941    | 3.40857   |

### Cooling
Bayesian Linear Regression                      
| Metric          | Training   | Test      |
|-----------------|------------|-----------|
| RMSE            | 3.13145    | 3.60074   |
| MAE             | 2.18106    | 2.80741   |

HMC Sampling for Linear Regression               
| Metric          | Training   | Test      |
|-----------------|------------|-----------|
| RMSE            | 3.14700    | 3.54940   |
| MAE             | 2.19436    | 2.69637   |

HMC for Classification                     
| Metric                    | Training   | Test |
|---------------------------|------------|------|
| Classification Error      | 1.30208%   | 0.0% |

Variational Inference                           
| Metric          | Training   | Test      |
|-----------------|------------|-----------|
| RMSE            | 3.13142    | 3.60122   |
| MAE             | 2.18110    | 2.80884   |
## References 

Data: Tsanas,Athanasios and Xifara,Angeliki. (2012). Energy Efficiency. UCI Machine Learning Repository. https://doi.org/10.24432/C51307.

Sampling code from the University of Bath, Bayesian Machine Learning CM50268 based on Radford Neal "one_step" code: http://www.cs.utoronto.ca/~radford/


