# Interpret OLS Regression Result

OLS (ordinary least square) 是一种基础的线性回归方法。本文的 “OLS result” 特指使用 `statsmodels.api.OLS` 得到的结果。

```py
import statsmodels.api as sm

data = sm.datasets.ccard.load_pandas()
X, y = data.exog.copy(), data.endog.copy()
X = sm.add_constant(X)
res = sm.OLS(y, X).fit()
res.summary()
```

运行以上代码可以得到一个表格：

![Inclusion-Exclusion Principle](images/OLS_summary.jpeg)

表格里是评估模型有效性的指标，但是解读起来需要一定的统计基础。本文将逐个解读其含义以及背后的统计学基础知识。

该表格分为了三个子表格，分别表达：

1. 评估模型显著性
2. 评估单个模型参数显著性
3. 评估模型偏差

我们将逐个表格讲解。

## Table 1: Model

跳过简单的字段。

### Df (Degree of freedom)

自由度是一个基础概念，在回归问题中，如果模型自由度=样本数，那么该回归问题退化到一个解方程的问题。因此我们一般希望有较高的 Df Residuals。自由度满足：

> No. Observations  = Df Residuals + Df Model

注意，当我们添加常数项，即：

```py
X = sm.add_constant(X)
```

因为常数项同样需要估计（也可以看作模型的一部分），该公式变为：

> No. Observations  = Df Residuals + Df Model + 1

### R-squared

R-squared 描述该模型对样本输入的拟合度。在本例中，其值为 0.244，即 24.4% 的变化可以被该模型解释。
它定义为：


$$ R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y})^2}{\sum_{i=1}^n (y_i - \bar{y})^2} $$

即，1 减去（模型误差平方和/以均值为预测值的误差平方和）。

```py
1.0 - res.ssr / res.centered_tss
```
可以通过这个代码检查，算出来 R-squared 是 `0.24357791682194052`，与表格相同。

直观来看，如果模型越准确，则 R-squared 值越接近 1.0。如果直接取样本输出的平均值为预测值，则 R-squared 为 0.0。如果模型甚至不如直接取样本的平均值准确，则 R-squared 可以为负值。

似乎，R-squared 越大，拟合效果越好。但是我们忽略了一个重要的因素，即上面提到的“自由度”。
只要我们往模型中增加参数，必然可以增加 R-squared，直到模型自由度与样本数相同，最终 R-squared = 1.0。但是这样得到的并不是一个有效的模型，而是存在明显的过拟合。

因此，我们需要引入 Adjusted R-squared。

#### Adj. R-squared

Adjusted R-squared 只会在新加入的模型变量有助于提高模型拟合度时增加，加入一个与结果无关的变量会降低它的数值。

它定义为：

$$\begin{aligned}
\bar{R}^2 &= 1 - \frac{SS_{res} / df_{res}}{SS_{tot} / df_{tot}} \\
    &= 1 - \frac{\sum_{i=1}^n (y_i - \hat{y})^2 / (n - p - 1)}{\sum_{i=1}^n (y_i - \bar{y})^2 / (n - 1)} \\
    &= 1 - (1 - R^2)\frac{n-1}{n - p - 1}
\end{aligned}$$

其中 p 是模型参数数量。

```py
1 - (res.ssr / (res.nobs - res.df_model - 1)) / (res.centered_tss / (res.nobs - 1))
```
可以通过这个代码检查，算出来 Adj. R-squared 是 `0.1984183894680266`，与表格相同。

根据公式，有两个推论：

1. Adj. R-squared 可以为负
2. Adj. R-squared <= R-squared

### F-statistic

F-statistic 衡量模型整体的显著性（后面要讲到的 t-statistic 是衡量单个模型参数的显著性）。

我们的原假设是：__模型所有参数为 0__。即，该模型是无效的，无论因子如何变化，都对结果不产生影响。

F-distribution 定义为两个 variance 的比值的分布：

$$ X = \frac{S_1 / df_1}{S_2 / df_2} $$

其中 $S_1$ 服从自由度为 $df_1$ 的 chi-squared 分布，$S_2$ 服从自由度为 $df_2$ 的 chi-squared 分布。

F-value 越大，说明模型比噪声越显著。即：

$$ \text{F-value} = \frac{\text{variance of y explained by model}}{\text{variance of y explained by error}} $$

```py
(res.ess / res.df_model) / (res.ssr / res.df_resid)
```
可以通过这个代码检查，算出来 F-value 是 `5.393721570932906`，与表格相同。

#### Prob(F-statistic)

F-distribution 由两个自由度的值确定，代入 F-statistic 可以得出其 p-value，就是这里的 Prob(F-statistic)。这个 p-value 的含义是原假设（__模型所有参数为 0__ ）成立情况下，出现当前结果的概率。它的 __值越小，说明这个模型越显著__。

### AIC and BIC


## Reference

1. [Interpreting OLS results](https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-statistics-toolbox/interpreting-ols-results.htm)

2. [Regression analysis basics](https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-statistics-toolbox/regression-analysis-basics.htm)
