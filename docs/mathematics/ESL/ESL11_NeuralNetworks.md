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
