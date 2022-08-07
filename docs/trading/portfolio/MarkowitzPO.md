# Portfolio Optimization with Known Parameters

首先引入一些之后使用的符号定义。

In practice, a portfolio deals with a whole universe of $N$ assets. We denote the log-returns of the $N$ assets at time index $t$ with the vector $\mathbf{r}_t \in \mathbb{R}$.

The time index $t$ can denote any arbitrary period such as minutes, days, weeks, etc. The historical data before $t$ is denoted by $\mathcal{F}_{t-1}$.

已知历史数据 $\mathcal{F}_{t-1}$，我们可以将回报率分解为 __期望回报率__ 和 __噪声__：

$$ \mathbf{r}_t = \mathbf{\mu}_t + \mathbf{\varepsilon}_t $$

其中，$\mathbf{\mu}_t$ 是根据历史数据 $\mathcal{F}_{t-1}$ 计算得到的期望收益：

$$\mathbf{\mu}_t = E(\mathbf{r}_t|\mathcal{F}_{t-1})$$

$\mathbf{\varepsilon}_t$ 是均值为 0 的白噪声，其方差为：

$$ \mathbf{\Sigma}_t = E[(\mathbf{r}_t - \mathbf{\mu}_t)(\mathbf{r}_t - \mathbf{\mu}_t)^T|\mathcal{F}_{t-1}] $$

我们通过历史数据主要是希望计算出 $\mathbf{\mu}_t$ 和 $\mathbf{\Sigma}_t$。

如果我们简单假设 $\mathbf{\mu}_t$ 和 $\mathbf{\Sigma}_t$ 不随时间变化，即 $\mathbf{r}_t$ 是独立同分布的，则利用 $1, ..., t-1$ 时刻的历史数据，可以简单估计：

$$ \mathbf{\mu}_t = \mathbf{\mu} = \frac{1}{t-1}\sum_{k=1}^{t-1} \mathbf{r}_k $$

$$ \mathbf{\Sigma}_t = \mathbf{\Sigma} = \frac{1}{t-2} \sum_{k=1}^{t-1} (\mathbf{r}_k - \mathbf{\mu})(\mathbf{r}_k - \mathbf{\mu})^T $$

假设我们总资金为 $B$ 美元，假设每种资产配置的权重为 $\mathbf{w}$，我们有：

- $B \mathbf{w}$ 表示在每种资产投入的资金，可以为负数，代表 short selling
- portfolio 的期望回报率是 $\mathbf{w}^T \mathbf{\mu}$
- portfolio 的风险（即波动率）为 $\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}$

夏普率(Sharpe Ratio) 表示每单位风险的超额收益：

$$ \text{SR} = \frac{\mathbf{w}^T \mathbf{\mu} - r_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}}$$

## 1 Markowitz Mean-Variance Portfolio Optimization

Markowitz mean-variance framework 是组合优化理论的基石。它的目标是寻找回报和风险的最佳 trade-off。我们可以将组合优化问题表示为：

$$\begin{align}
\mathop{\arg \max}_{\mathbf{w}} & \mathbf{w}^T \mathbf{\mu} - \lambda \mathbf{w}^T\mathbf{\Sigma}\mathbf{w} \\
\text{s.t.} & \mathbf{w} \mathbf{\mu} >= \mu_0, \\
& \mathbf{1}^T \mathbf{w} = 1
\end{align}$$

其中，$\lambda$ 参数反映投资者的风险厌恶程度。

该优化问题可以简单用 Lagrange multiplier 解决，令：

$$ L(\mathbf{w}, \alpha) = \mathbf{w}^T \mathbf{\mu} - \lambda \mathbf{w}^T\mathbf{\Sigma}\mathbf{w} + \alpha (\mathbf{1}^T\mathbf{w} - 1) $$

解方程组：

$$\begin{cases}
\dfrac{\partial L}{\partial \mathbf{w}} &= \mathbf{\mu} - 2\lambda \mathbf{\Sigma}\mathbf{w} + \alpha\mathbf{1} = 0 \\
\dfrac{\partial L}{\partial \alpha} &=  \mathbf{1}^T\mathbf{w} - \mathbf{1} = 0
\end{cases}$$

从 1 式得到：

$$ \mathbf{w} = \frac{1}{2\lambda} \mathbf{\Sigma}^{-1}(\mathbf{\mu} + \alpha\mathbf{1})$$

将 $\mathbf{w}$ 带入 2 式，得到上式的 $\alpha$：

$$ \alpha = \dfrac{2\lambda - \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{\mu} }{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}$$

至此，该优化问题的最优解确定。

## Reference

1. [Portfolio Optimization with Known Parameters](https://palomar.home.ece.ust.hk/papers/2016/Feng&Palomar-FnT2016.pdf)
2. [Asset Allocation using Convex Portfolio Optimization](https://medium.com/where-quant-meets-data-science/asset-allocation-using-convex-portfolio-optimization-f47398d4d613)
