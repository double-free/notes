# ESL 11: Neural Networks

神经网络的中心思想是将输入 __线性组合__ 为一些衍生的特征，再建立输出与这些特征之间的 __非线性__ 模型。

## 11.2 Projection Pursuit Regression

以一个通用的监督学习问题为例，假设我们有 $p$ 维输入 $X$，输出是 $Y$。$w_m$ 是 $p$ 维单位向量，我们可以把 projection pursuit regression 模型表示为：

$$f(X) = \sum_{m=1}^M g_m (w_m^T X) $$

可以看出，这也是一个加性模型。但是区别在于，它的自变量不是直接输入 $X$，而是输入的线性组合 $w_m^T X$。

$g_m(w_m^T X)$ 被称为“岭函数” (Ridge Function)。它只沿着 $w_m$ 的方向变化，而标量 $V_m = w_m^T X$ 就是输入 $X$ 在 $w_m$ 方向上的投影 (projection)。

我们的目标是寻找 $w_m$ （即投影方向）使得模型估计误差最小。因此，这个方法叫做 projection pursuit。它的 __优点__ 是如果子模型数量 M 足够大，它能够完美拟合任何连续函数。__缺点__ 是可解释性差。因此适用于只需要做预测，不需要归因的场景。

### PPR 拟合

给定训练数据 $(x_i, y_i), i=1,2,\dots,N$，我们的目标是确定函数 $g$ 和方向 $w$ ，使预测结果的 squared error 最小：

$$ g, w = \mathop{\arg \min}_{g, w} \sum_{i=1}^N [y_i - \sum_{m=1}^M g_m(w_m^T x_i)]^2 $$

假设仅有一个子模型，即 __M = 1__，确定 $g$ 的过程其实就是一个一维 smoothing 问题。因此，$g$ 可以选择使用 spline。

已知函数 $g$ 的形式，我们需要确定使估计误差最小的方向 $w$。这是一个 __无约束的优化问题__，且 $g$ 可导，因此可以使用牛顿法来解决。

假设当前对 $w$ 的估计为 $w_{\text{old}}$，我们对 $g$ 进行泰勒展开，忽略 2 阶以上有：

$$ g(w^T x_i) \approx g(w_{\text{old}}^T x_i) + g'(w_{\text{old}}^T x_i)(w - w_{\text{old}})^T x_i $$

由于 M = 1，squared error 可以简化为：

$$\begin{align}
\sum_{i=1}^N [y_i - g(w^T x_i)]^2 &= \sum_{i=1}^N [y_i - g(w_{\text{old}}^T x_i) - g'(w_{\text{old}}^T x_i)(w - w_{\text{old}})^T x_i]^2 \\
&= \sum_{i=1}^N g'(w_{\text{old}}^T x_i)^2 [w^T x_i - (w_{\text{old}}^T x_i + \dfrac{y_i - g(w_{\text{old}}^T x_i)}{g'(w_{\text{old}}^T x_i)})]^2
\end{align}$$

等式右边可以看作一个 least squares regression 问题。有 N 个样本点，对于第 i 个样本，其平方误差的权重为 $g'(w_{\text{old}}^T x_i)^2$ ，目标是 $w_{\text{new}}^T x_i$ 尽量靠近 $w_{\text{old}}^T x_i + \frac{y_i - g(w_{\text{old}}^T x_i)}{g'(w_{\text{old}}^T x_i)}$ 。


求解这个 least squares regression 我们得到一组新的系数 $w_{\text{new}}$，更新 $w_{\text{old}} = w_{\text{new}}$ 并进行下一轮迭代，直到 $g'(w_{\text{old}}^T x_i)$ 小于某个阈值。

由于其计算量过大，PPR 的应用并不很广泛。但是，它是后来获得广泛应用的 __神经网络技术的前身__。我们将在下面的章节介绍神经网络。

## 11.3 Neural Networks

“神经网络”这个名字源于该方法最早被应用于人脑的建模。每个节点是一个神经元，他们之间的连接代表突触。单“隐藏层”的神经网络与刚才介绍的 Projection Pursuit Regression 模型非常相似，我们以它为例讲解。

![Neural Network](images/11/neural_network.png)

我们可以看到该神经网络分为 3 层。其中 $X$ 是输入，$Y$ 是输出，$Z$ 是所谓的“隐藏层”，它被称为隐藏层是因为它不直接可见。

类似于 PPR，隐藏层 $Z$ 由输入 $X$ 线性组合，再附加一个“激活函数” $\sigma$ 得出。

$$ Z_m = \sigma(\alpha_0 + \alpha_m^T X), \quad m = 1,\dots,M $$

常见的激活函数如 sigmoid：

$$ \sigma(v) = \dfrac{1}{1 + e^{-v}} $$

而输出 $Y$ 由隐藏层 $Z$ 线性组合，再附加一个“输出函数” $g$ 得出。

$$ Y_k = g_k(T) = g_k(\beta_0 + \beta_k^T Z) $$

对于回归问题，$g$ 可以省略，对于分类问题，为确保输出都是整数且和为 1.0，通常选用 softmax 函数，属于第 k 类的概率为:

$$ g_k(T) = \dfrac{e^{T_k}}{\sum_{l=1}^K e^{T_l}} $$

我们看出，其实 PPR 与 NN 的差异就在于 NN 使用的激活函数相比 PPR 使用的 spline 简单很多，这就使得 NN 的计算量小很多，获得了更广泛的应用。

## 11.4 Fitting Neural Networks

拟合神经网络实际上就是找到刚才提到的两组参数：

1. 由输入 $X$ 到隐藏层 $Z$ 的线性组合系数 $\bf{\alpha}$。由于有 M 个隐藏层节点，而每个节点对应的系数都是 $p + 1$ 维（+1 for bias）。因此是一个 $M \times (p+1)$ 矩阵。

2. 由隐藏层 $Z$ 到输出 $Y$ 的线性组合系数 $\bf{\beta}$。由于有 K 个输出节点，而每个节点对应的系数都是 $M + 1$ 维（+1 for bias）。因此是一个 $K \times (M+1)$ 矩阵。

我们首先以损失函数 sum-of-squared-errors 为例:

$$ R(\theta) = \sum_{i=1}^N R_i = \sum_{i=1}^N \sum_{k=1}^K (y_{ik} - f_k(x_i))^2 $$

记 $l, m, k$ 分别为输入 $X$，隐藏层 $Z$ 和输出 $Y$ 的序号，对 $\alpha_{ml}$ 和 $\beta_{km}$ 求导，根据链式法则有：

$$ \dfrac{\partial R_i}{\partial \beta_{km}} = - 2(y_{ik} - f_k(x_i)) g_k'(\beta_k^T z_i) z_{mi} $$

$$ \dfrac{\partial R_i}{\partial \alpha_{ml}} = -  \sum_{k=1}^K 2(y_{ik} - f_k(x_i)) g_k'(\beta_k^T z_i) \beta_{km} \sigma'(\alpha_m^T x_i) x_{il} $$

注意，对 $\beta_{km}$ 求导结果中不含 $\sum_{k=1}^K$。我们假设现在用第 $j(j \neq k)$ 个输出对 $\beta_{km}$ 求导，由于 $\beta_{km}$ 意义是第 k 个输出与第 m 个隐藏节点的系数，与第 j 个输出无关，结果必然为 0。而对于 $\alpha_{ml}$ 求导时，由于第 m 个隐藏节点会作用给第 k 个和第 j 个输出，所以存在 $\sum_{k=1}^K$。

得到这些导数，我们可以设定 learning rate $\gamma_r$，使用梯度下降迭代更新：

$$ \beta_{km}^{(r+1)} = \beta_{km}^{(r)} - \gamma_r \sum_{i=1}^N \dfrac{\partial R_i}{\partial \beta_{km}^{(r)}} $$

$$ \alpha_{ml}^{(r+1)} = \alpha_{ml}^{(r)} - \gamma_r \sum_{i=1}^N \dfrac{\partial R_i}{\partial \beta_{ml}^{(r)}} $$

现在令：

$$ \dfrac{\partial R_i}{\partial \beta_{km}} = \delta_{ki} z_{mi} $$

$$ \dfrac{\partial R_i}{\partial \alpha_{ml}} = s_{mi} x_{il} $$

我们可以得出关系：

$$ s_{mi} = \sigma'(\alpha_m^T x_i) \sum_{k=1}^K \beta_{km} \delta_{ki} $$

这个等式被称为 back-propagation equation。利用这个等式，更新时可以简化 $s_{mi}$ 的计算。back-propagation 的过程可以描述为一个 __双向传播__ 的过程：

1. 正向传播，利用输入 $X$ 和当前的 weights 来计算预测值 $\hat{f}_k(x_i)$

2. 反向传播，首先计算出隐藏层 $Z$ 到输出层 $Y$ 的 $\delta_{ki}$，再利用上面的等式计算 $s_{mi}$，得出两个梯度的值，再更新 weights

这个算法的优势在于简单并且易于并行，劣势在于计算量大。
