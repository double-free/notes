# 向量和矩阵求导

向量、矩阵求导其实就两个内容

1. 分子每个元素对分母每个元素求导
2. 将结果以一定方式布局

对于 1，没什么特别的，就是标量之间的求导。

对于 2，我们需要分情况讨论。


## 求导布局

求导结果的布局根据定义不同有所不同，没有统一。所以经常在不同的书上看到不一样的公式，使人产生困惑。

常见的求导类型如下：

| 分母 \ 分子 | 标量 | 向量 | 矩阵 |
| :-: | :-: | :-: | :-: |
| 标量 | $\dfrac{\partial y}{ \partial x}$| $\dfrac{ \partial \textbf{y} }{ \partial x }$ | $\dfrac{\partial \textbf{Y}}{\partial x}$ |
| 向量 | $\dfrac{\partial y}{ \partial \textbf{x}}$ | $\dfrac{\partial  \textbf{y} }{ \partial \textbf{x}}$ | / |
| 矩阵 |$\dfrac{ \partial y }{ \partial \textbf{X} }$ | / | / |

我们划掉的类型是因为其结果无法在二维矩阵中很好地表示，在优化问题中也不常见。

未划掉的类型中，唯一布局有歧义的就是向量对向量的求导：$\dfrac{ \partial  \textbf{y} }{ \partial \textbf{x} }$

## 向量对向量求导

歧义在于，假设 $\textbf{y}$ 是一个 $m$ 维向量，$\textbf{x}$ 是一个 $n$ 维向量，那求导结果是一个 $m \times n$ 矩阵还是 $n \times m$ 矩阵呢？

- 分子布局，即以分子 $\textbf{y}$ 的元素数作为行数。结果是一个 $m \times n$ 矩阵，也称为雅可比（Jacobian）矩阵。

$$ \frac{ \partial \textbf{ y } }{ \partial \textbf{ x } } = \begin{bmatrix}
 \frac{ \partial {y_1} }{ \partial  {x_1} } & \frac{\partial {y_1} }{\partial  {x_2} } & \cdots &\frac{\partial {y_1} }{\partial  {x_n} }  \\
 \frac{\partial {y_2} }{\partial  {x_1} } & \frac{\partial {y_2} }{\partial  {x_2} } & \cdots &\frac{\partial {y_2} }{\partial  {x_n} }  \\
 \vdots & \vdots & & \vdots \\
  \frac{\partial {y_m} }{\partial  {x_1} } & \frac{\partial {y_m} }{\partial  {x_2} } & \cdots &\frac{\partial {y_m} }{\partial  {x_n} }  \\
\end{bmatrix}_{m \times n} $$

- 分母布局，即以分母 $\textbf{x}$ 的元素数作为行数。结果是一个 $n \times m$ 矩阵，也称为梯度(Gradient)矩阵。

$$ \frac{\partial \textbf{ y }}{\partial \textbf{ x } } = \begin{bmatrix}
 \frac{\partial {y_1} }{\partial  {x_1} } & \frac{\partial {y_2} }{\partial  {x_1} } & \cdots &\frac{\partial {y_m} }{\partial  {x_1} }  \\
 \frac{\partial {y_1} }{\partial  {x_2} } & \frac{\partial {y_2} }{\partial  {x_2} } & \cdots &\frac{\partial {y_m } }{\partial  {x_2} }  \\
 \vdots & \vdots & & \vdots \\
  \frac{\partial {y_1} }{\partial  {x_n} } & \frac{\partial {y_2} }{\partial  {x_n} } & \cdots &\frac{\partial {y_m} }{\partial  {x_n} }  \\
\end{bmatrix}_{n \times m} $$

两种布局均可，在一本书中一般是一致的。

## 标量对向量求导

标量常见的有以下几种形式：

1. $a^T x$
2. $x^T a$
3. $x^T A x$

从定义上看，1 和 2 类似：

首先定义：

$$S = a^T x = x^T a  = \sum_{i=1}^n a_ix_i$$

得出：

$$ \frac{\partial S}{\partial x_i} = a_i$$

因此：

$$\frac{\partial a^Tx}{\partial x} = \frac{\partial x^Ta}{\partial x} = [ \frac{\partial S}{\partial x_1}, \frac{\partial S}{\partial x_2}, \cdots, \frac{\partial S}{\partial x_n}]^T = a$$

3 稍微复杂：

$$ S = \sum_{i=1}^n \sum_{j=1}^n x_iA_{i,j}x_j$$

$$ \frac{\partial S}{\partial x_k} = \sum_{j=1}^n A_{k,j}x_j + \sum_{i=1}^n x_iA_{i,k} = (A_{k,i} + A_{i,k})x_i$$

即求导后向量的第 k 个元素是 A 的第 k 行与 x 的内积 + 第 k 列与 x 的内积。这其实就是矩阵与向量乘法的定义。

$$\frac{\partial x^TAx}{\partial x} = [ \frac{\partial S}{\partial x_1}, \frac{\partial S}{\partial x_2}, \cdots, \frac{\partial S}{\partial x_n}]^T = Ax + A^Tx $$

### 例：最小二乘法

最小二乘法是最流行的线性模型拟合方法。它的目的是找出系数 $\mathbf{\beta}$ 使 $||Y-\hat Y||_2$ （residual sum of squares, RSS）最小：

$$\text{RSS}(\mathbf{\beta} ) = \sum_{j=1}^N (y_j - X_j^T\mathbf{\beta} )^2 $$

其中 $j$ 代表训练数据的序号。一共有 $N$ 组训练数据。
用矩阵形式表示为：

$$\text{RSS}(\mathbf{\beta}) = (\textbf{y} - \textbf{X}\mathbf{\beta} )^T(\textbf{y} - \textbf{X}\mathbf{\beta} )$$


 这里需要用 $\text{RSS}(\mathbf{\beta})$ 对 $\mathbf{\beta}$ 求导，得出二次函数最值点。

$$\text{RSS}(\mathbf{\beta}) = \textbf{y}^T\textbf{y} -\textbf{y}^T \textbf{X} \mathbf{\beta} - \mathbf{\beta}^T \textbf{X}^T \textbf{y} + \mathbf{\beta}^T \textbf{X}^T \textbf{X}\mathbf{\beta}$$

套用上面的结论，可以得到：

$$ \frac{ \partial \text{RSS}(\mathbf{\beta})}{\partial \mathbf{\beta}} = - 2\textbf{X}^T\textbf{y} + 2\textbf{X}^T\textbf{X}\mathbf{\beta}$$

令其为 0 可以解出：

$$\hat{\mathbf{\beta}} = (\textbf{X}^T \textbf{X})^{-1} \textbf{X}^T \textbf{y}$$
