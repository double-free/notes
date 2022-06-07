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



## 4.4 Logistic regression


## Reference

1. [masking in linear regression for multiple classes](https://stats.stackexchange.com/questions/475458/masking-in-linear-regression-for-multiple-classes)
