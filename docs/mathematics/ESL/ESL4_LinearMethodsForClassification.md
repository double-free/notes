# ESL 4: Linear Methods for Classification

## 4.2 Linear Regression of an Indicator Matrix

对于分类问题，我们可以把每个分类用 indicator variable 来编码。

例如，如果有 $K$ 个类型，我们可以定义：

$$ \textbf{y} = [y_1, ..., y_K]^T $$

当样本是第 $i$ 类时，$y_i = 1$，其它值为 0。

当我们有 N 个样本时，可以写成一个 $N \times K$ 矩阵（indicator response matrix）：

$$ \textbf{Y} = [\textbf{y}_1, ..., \textbf{y}_N]^T $$

这个矩阵每一行仅有一个 1，其余值为 0。这个 1 代表该行样本的类型。

这实际上就是 one-hot 编码，这种编码相对于直接使用整数 $1, ..., K$ 来表示的有优势在于 __类型间的距离都是一样的__。

利用第三章的 Linear Regression，我们可以得出系数矩阵 $\hat{\textbf{B}}$：

$$ \hat{\textbf{B}} = \textbf{X}(\textbf{X}^T \textbf{X})^{-1} \textbf{X}^T \textbf{Y} $$

$\hat{\textbf{B}}$ 是一个 $(p+1) \times K$ 的矩阵。$\textbf{X}$ 是 $N \times (p+1)$ 矩阵。

得到估计值：

$$ \hat{\textbf{Y}} = \textbf{X}(\textbf{X}^T \textbf{X})^{-1} \textbf{X}^T \textbf{Y}$$

该方法的步骤是：

1. 对类型进行 one-hot 编码得到 $\textbf{Y}$
2. 将问题视作一个多变量的线性回归问题，对 one-hot 编码后的每个 bit 进行拟合得到 $\hat{\textbf{B}}$
3. 在判断类别时，选择 $\textbf{X}\hat{\textbf{B}}$ 每行的最大值所表示的类型

使用 Linear Regression 解决分类问题的最大问题在于，当类别数 $K \leq 3$ 时，可能会出现某个类被掩盖（mask）的情况。其本质原因是 Linear Regression 的“刚性”特质，即它的分界面是不够灵活的。

举个简单的例子，对于以下三个 1 维正态分布：

- class1: $x \sim N(1, 1)$
- class2: $x \sim N(5, 1)$
- class3: $x \sim N(9, 1)$

分布如下：

![Distribution](images/4/masking1.png)

如果我们用 Linear Regression 拟合，那我们可以得到 3 组 $\beta_0, \beta_1$，分别对应第 $i$ 组，可以用来计算 $y_i$ 的值：

$$ y_i = \beta_0 + \beta_1 x $$

![Classification](images/4/masking2.png)

可以看到，$y_1$ （对应class2）从来不是最大值。也就是说我们的分类结果中只有 class1 和 class3 了，class2 被 mask 了。可以通过结果验证：

```py
pd.DataFrame(X*B).T.idxmax().value_counts()
# 2    1505
# 0    1495
# dtype: int64
```
代码：

```py
import numpy as np
import pandas as pd

def generate_nd_sample(name, mu_array, sigma, N):
    xx = {}
    for i in range(1, len(mu_array) + 1):
        xi = np.random.normal(mu_array[i-1], sigma, N)
        xx[f"x{i}"] = xi
    xx["name"] = name
    return pd.DataFrame(xx).astype({"name":"category"})

s1 = generate_nd_sample("class1", [1], 1, 1000)
s2 = generate_nd_sample("class2", [5], 1, 1000)
s3 = generate_nd_sample("class3", [9], 1, 1000)

s = pd.concat([s1, s2, s3]).reset_index(drop=True)

# Linear Regression

tmp = s.copy()
tmp.insert(0, "ones", np.ones(s.shape[0]))
X = np.matrix(tmp.drop("name", axis=1).to_numpy().T).T

Y = np.matrix(pd.get_dummies(s.name))

def LR_beta(X, Y):
    # class count
    K = Y.shape[1]
    return (X.T * X).I * X.T * Y

B = LR_beta(X, Y)

# Plot

from bokeh.io import output_notebook
output_notebook()
from bokeh.palettes import viridis
from bokeh.plotting import figure, show
import itertools

def create_histogram_figure(sample_data):
    # sample data must be like:
    # | x | category |
    assert sample_data.shape[1] == 2, "input must be of shape (N, 2)"

    x = sample_data.iloc[:, 0]
    categories = sample_data.iloc[:, 1].unique()

    fig = figure(x_axis_label=x.name, y_axis_label="counts")

    color_gen = itertools.cycle(viridis(len(categories)))
    for (category, color) in zip(categories, color_gen):
        data = sample_data[sample_data.iloc[:, 1] == category]
        counts, bins = np.histogram(data.iloc[:, 0], bins='auto')
        fig.quad(top=counts, bottom=0, left=bins[:-1], right=bins[1:],
                   alpha=0.5, color=color, legend_label=str(category))

    return fig

def create_line_figure(lines, x_start, x_end):
    fig = figure(x_axis_label="x", y_axis_label="y")
    color_gen = itertools.cycle(viridis(lines.shape[0]))
    for i in range(lines.shape[0]):
        b0, b1 = lines[i,0], lines[i, 1]
        fig.line(x=[x_start, x_end], y = [b0 + b1*x_start, b0 + b1*x_end],
                 line_width=2, alpha=0.5, color=next(color_gen),
                 legend_label=f"y{i} = {b0:.6f} + {b1:.6f}x")
    return fig

hist_fig = create_histogram_figure(s)
line_fig = create_line_figure(B.T, s.x1.min(), s.x1.max())
from bokeh import layouts
show(layouts.column(hist_fig, line_fig))
```

## 4.3 Linear discriminant analysis

为了获得最优的分类结果，我们需要知道后验概率（$X = x$ 时属于第 $k$ 类的概率）：

$$ \text{Pr}(G=k | X=x) = \frac{f_k(x) \pi_k}{\sum_{i=1}^K f_i(x) \pi_i}$$

因为本质上，我们是在找到一个 k 使得后验概率最大，即：

$$\begin{align}
\hat{k} &= \mathop{\arg \max}_{k}  \text{Pr}(G=k | X=x) \\
&= \mathop{\arg \max}_{k}  f_k(x) \pi_k \\
&= \mathop{\arg \max}_{k}  [\ln(f_k(x)) + \ln(\pi_k)]
\end{align}$$

这被称为 __判别函数__（discriminant function），其中：

- $f_i(x)$ 是第 i 类样本取 x 的概率
- $\pi_i$ 是属于第 i 类的先验概率

这里的难点在于确定 $f_i(x)$，显然 $\pi_i$ 的估计是可以通过样本数据直接得到的。

线性判别分析（Linear Discriminant Analysis, LDA）__假设变量 X 服从多维高斯分布__（X 包含多维）：

$$ f_k(x) = \frac{1}{(2 \pi)^{p/2} |\mathbf{\Sigma}_k|^{1/2}} e^{-\frac{1}{2}(x - \mu_k)^T \mathbf{\Sigma}_k^{-1} (x - \mu_k)} $$

带入最优分类的式子, 逐步去掉与 $k$ 无关的部分：

$$\begin{align}
\hat{k} &= \mathop{\arg \max}_{k}  [\ln(f_k(x)) + \ln(\pi_k)] \\
&= \mathop{\arg \max}_{k}  [- \ln((2 \pi)^{p/2} |\mathbf{ \Sigma }_k|^{1/2}) - \frac{1}{2}(x - \mu_k)^T \mathbf{ \Sigma }_k^{-1} (x - \mu_k) + \ln(\pi_k)] \\
&= \mathop{\arg \max}_{k}  [- \frac{1}{2} \ln |\mathbf{ \Sigma }_k|^{1/2} - \frac{1}{2}(x - \mu_k)^T \mathbf{\Sigma}_k^{-1} (x - \mu_k) + \ln(\pi_k)] \\
\end{align}$$

此时，判别函数为：

$$\delta_k(x) = - \frac{1}{2} \ln |\mathbf{ \Sigma }_k|^{1/2} - \frac{1}{2}(x - \mu_k)^T \mathbf{\Sigma}_k^{-1} (x - \mu_k) + \ln(\pi_k)$$

是 $x$ 的二次函数。因此称为二次判别分析(Quadratic Discriminant Analysis, QDA)。

我们再 __假设每个类中变量 X 分布的方差是相等的__，则 $\mathbf{\Sigma}$ 也与 $k$ 无关了，可以进步一化简判别函数为：

$$\delta_k(x) = x^T\mathbf{\Sigma}^{-1}\mu_k - \frac{1}{2}\mu_k^T\mathbf{\Sigma}^{-1} \mu_k + \ln(\pi_k)$$

我们可以看出，化简后判别函数对于 $x$ 是 __线性__ 的。这说明两个类的分界面（即判别函数相等时）也是线性的。因此叫做线性判别分析(Linear Discriminant Analysis, LDA)。

实际中，我们可以通过样本估计高斯分布的参数：

- $\hat{\pi}_k = N_k / N$，即第 k 类的样本数占总样本数的比例
- $\hat{\mu}_k = \sum_{g_i = k} x_i / N_k$，即第 k 类样本 X 的平均值
- $\hat{\mathbf{\Sigma}} = \sum_{k=1}^K \sum_{g_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T / (N - K)$，对协方差矩阵的无偏估计，证明在 ESL3 中

有了判别函数的表达式 $\delta_k(x)$，我们只需要依次带入 $k = 1, ..., K$, 当得到的 $\delta_k(x)$ 最大时的 $k$ 即为最佳分类。


### 4.3.2 Computation of LDA

协方差矩阵 $\mathbf{\Sigma}$ 是一个对称矩阵，可以进行特征值分解：

$$ \mathbf{\Sigma} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T $$

其中：$\mathbf{Q}$ 是单位正交矩阵，$\mathbf{\Lambda}$ 是对角阵。带入判别函数有：

$$\begin{align}
\delta_k(x) &= x^T\mathbf{\Sigma}^{-1}\mu_k - \frac{1}{2}\mu_k^T\mathbf{\Sigma}^{-1} \mu_k + \ln(\pi_k) \\
&= x^T\mathbf{Q}^{T}\mathbf{\Lambda}^{-1}\mathbf{Q}\mu_k - \frac{1}{2}\mu_k^T\mathbf{Q}^{T}\mathbf{\Lambda}^{-1}\mathbf{Q}\mu_k + \ln(\pi_k)
\end{align}$$

令：

$$ x^{*} = \mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{Q}x $$

$$ \mu^{*} = \mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{Q} \mu $$

有：

$$ \delta_k(x^{*}) = x^{* T}\mu_k^{*} - \frac{1}{2}\mu_k^{* T} \mu_k^{*} + \ln(\pi_k) $$

当我们判断某个样本 $x_1$ 属于 m 和 n 中的哪一个类时，可以比较其判别函数，我们判断它是 m 类如果满足：

$$ \delta_m(x_1^*) > \delta_n(x_1^*) $$

带入表达式有：

$$ x^{*T} (\mu^*_m - \mu^*_n) > \frac{1}{2} (\mu^*_m + \mu^*_n)^T(\mu^*_m - \mu^*_n) - \ln(\pi_m/\pi_n)$$

这样看起来就非常直观了。 **LDA 是将样本投影在两个类中心的连线上，并且比较它更靠近哪一边，以此决定它属于哪个类**。当然，这个过程还考虑了两个类的先验概率（$\ln(\pi_m/\pi_n)$ 项）。


### 4.3.3 Reduced-Rank Linear Discriminant Analysis

LDA 也是一种降维的手段。假设我们有 $p$ 维特征，$K$ 个类别。根据 4.3.2 中介绍的计算方式，我们一共有 $K$ 个类中心点。他们一定在一个最高 $K-1$ 维的空间里。

例如，对于 2 个类的分类问题，__无论特征是多少维__，我们只有 2 个类中心点。他们必定在一条直线（1维）上。同理，对于 3 个类的分类问题，我们只有 3 个类中心点。他们必定在一个平面（2维）内，如果特征维度大于等于 2。

因此，经过 LDA，原始数据总能被投影到一个超平面上，其维度为（对应 sklearn LDA 方法中的 `n_components` 参数）：

$$ \text{n_components} = \min(p, K-1) $$

这说明，__在 $p \gg K$ 时，使用 LDA 可以将一个 $p$ 维的输入降维到 $K-1$ 维__。

我们以 sklearn 中的 wine 数据集为例。它具有 13 维特征，3 个类别。我们使用 LDA 可以将这些数据投影到一个 2 维的平面上。

![LDA projection of wine data](images/4/lda.png)

代码：

```py
import pandas as pd
import numpy as np
from sklearn import datasets

wine = datasets.load_wine()
X = pd.DataFrame(wine.data, columns = wine.feature_names)
y = wine.target

# LDA projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
model.transform(X)

# plot
data_to_plot = pd.DataFrame(
    np.insert(model.transform(X), 2, y, axis=1),
    columns=["x1", "x2", "class"])
show(create_scatter_figure("LDA projection for wine data", data_to_plot))
```


## 4.4 Logistic regression


## Reference

1. [masking in linear regression for multiple classes](https://stats.stackexchange.com/questions/475458/masking-in-linear-regression-for-multiple-classes)
2. [Linear discriminant analysis, explained](https://yangxiaozhou.github.io/data/2019/10/02/linear-discriminant-analysis.html)
