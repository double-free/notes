# Wiener processes and Ito’s lemma

若某一变量的以一种不确定的方式随时间变化，我们称它服从某种随机过程（stochastic process）。随机过程可以分为离散时间和连续时间两类。而变量本身也可以分为连续变量和离散变量两类。

本章我们将导出股票价格的连续变量，连续时间的随机过程。

## 13.1 马尔科夫性质

Markov Process 是一种特殊的随机过程。在该过程中，只有当前的值与未来的预测相关。

我们通常假设股票符合一个 Markov Process。即我们对股票未来价格的预测只和股票当前价格有关，与一周前，一年前的价格无关。

这也符合弱型市场有效性（the weak form of market efficiency）。它指出，一种股票的当前价格包含过去价格的所有信息。如果弱型市场有效性不成立，则分析师可以通过历史数据获得高于平均收益率的收益。事实上，现实中我们没有任何证据证明可以做到这一点。

## 13.2 连续时间随机过程

假设一个变量服从 Markov Process。它现在的值是 10，在一年中的变化满足正态分布 $N(\mu, \sigma^2)$。那么它在 2 年中的变化的概率分布是什么？

假设在第 1 年中的变化是 $x_1$，第二年中的变化是 $x_2$。假设两者相互独立。

显然，$x_1$和 $x_2$ 都服从 $N(\mu, \sigma^2)$。则两者的和服从分布 $N(2\mu, 2\sigma^2)$。

同理，0.5 年中的变化的概率分布为 $N(0.5\mu, 0.5\sigma^2)$。

结论是，变量在任意时间段 $T$ （以年为单位）变化的分布服从 $N(\mu T, \sigma^2 T)$。

#### 证明

均值部分很简单，可以稍微证明一下方差部分。

已知 $D(x_1)$ 和 $D(x_2)$，且$x_1$，$x_2$ 相互独立，求$D(x_1 + x_2)$ 。

将 $D(x_1 + x_2)$ 做如下变形：

$$\begin{align}
D(x_1 + x_2) &= E[(x_1+x_2)^2] - [E(x_1+x_2)]^2 \\
&= D(x_1) + D(x_2) + 2E(x_1 x_2) - 2E(x_1)E(x_2)
\end{align}$$

由于 $x_1$，$x_2$ 相互独立，有：

$$COV(x_1,x_2) = E(x_1 x_2) - E(x_1)E(x_2) = 0$$

因此有：

$$D(x_1 + x_2) = D(x_1) + D(x_2)$$

### 12.2.1 维纳过程

如果我们令以上变化的期望 $\mu = 0$，$\sigma = 1$，我们就得到 __维纳过程__(Wiener Process)。它在物理学中被用来描述某个粒子受到大量分子碰撞的运动，也被称作 __布朗运动__(Brownian Motion)。

严格来讲，一个服从维纳过程的变量 $z$ 具有如下两条性质：

1. 变化量 $\Delta z$ 在一个小的时间 $\Delta t$ 中符合：

$$\Delta z = \epsilon\sqrt{\Delta t}$$

其中$\epsilon$ 服从标准正态分布 $N(0,1)$

2. 变化量 $\Delta z$ 在任意两个不同的时间段独立

#### 广义维纳过程

目前我们讨论的维纳过程中，对 $z$ 未来时刻的期望总是等于它的初值。我们将其做一定推广，即对其叠加一个随时间变动的因子$a~dt$：

$$dx = a~dt + b~dz$$

其中，$a~dt$ 表示 $x$ 单位时间内漂移速度为 $a$。
$b~dz$ 表示噪声，该噪声的变动是 $b$ 倍的维纳过程，即服从$N(0, b^2)$。

在一个很短的时间 $\Delta t$ 中，$x$ 的变动 $\Delta x$ 可以写为：

$$\Delta x = a \Delta t + b \epsilon \sqrt{\Delta t}$$

因此，在 $T$ 时刻它服从正态分布 $N(aT, b^2T)$。

一个典型的广义维纳过程如下所示：
![Generalized Wiener process](https://upload-images.jianshu.io/upload_images/4482847-5661618f0705732d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 例

> 一个公司的现金服从广义维纳过程，drift 为 0.5 每季度，方差是 4.0 每季度。该公司需要多少初始现金才能保证 1 年后现金为负的概率小于 5%？

根据上面广义维纳过程的推论，假设其初始现金是$m_0$，1 年后现金 $m$ 服从 $N(m_0 + 4 \times 0.5, 4 \times 4)$。

假设 $\epsilon$  服从标准正态分布 $N(0,1)$，显然 $m = m_0 + 2 + 4\epsilon$。

因此问题转化为 $P(m < 0) = P(\epsilon < -\frac{m_0+2}{4})\leq 0.05$。

查[标准正态分布表](https://wenku.baidu.com/view/cd811b103a3567ec102de2bd960590c69fc3d849.html)得到 $P(\epsilon < 1.65) = 0.9505$，因此 $P(\epsilon < -1.65) = 0.0495$。因此有$m_0 = 4.6$。即需要初始资金 4.6 百万美元。

### 12.2.2 伊藤过程

当广义维纳过程中的 $a$，$b$ 不是常数，而是关于$(x,t)$ 的函数时，就变为了一个伊藤过程：

$$dx = a(x,t)~dt + b(x,t)~dz$$

在一个很短的时间 $\Delta t$ 内，假设 $a$ 和 $b$ 在 $t$ 到 $t+\Delta t$中保持不变，我们有：

$$\Delta x = a(x, t)~\Delta t + b(x,t) \epsilon \sqrt{\Delta t}$$

## 12.3 描述股价变化的过程

假设我们对股票的期望收益率为 $\mu$，则经过一个很短的时间 $\Delta t$，我们期望的股票价值是 $\mu S \Delta t$。利用伊藤过程来理解，则股票价格的漂移速度 (drift rate) 是 $\mu S$。

如果我们不考虑任何扰动，即股票价格总按照期望上涨。则可以计算：

$$\frac{dS}{dt} = \mu S$$

将 S 从 0 至 T 积分，可以得出：

$$S_T = S_0 e^{\mu T}$$

现实中肯定是存在扰动的，我们一般假设扰动与股票价格成正比，假设比例是$\sigma$，则我们得到以下伊藤过程：

$$dS = \mu S dt + \sigma S dz $$

即：

$$\frac{ dS}{ S} = \mu dt + \sigma dz$$

其离散时间模型为：

$$\frac{\Delta S}{S} = \mu \Delta t + \sigma \epsilon \sqrt{\Delta t}$$

可以发现 $\frac{ \Delta S }{ S }$ 服从正态分布 $N(\mu \Delta t, \sigma^2 \Delta t)$。

其中：
- $\mu$：股票的期望回报率，在风险中性假设下，$\mu$ 等于无风险利率 $r$
- $\sigma$:  股票价格的波动率

#### 例
>分析以上股票价格变化过程与下面三个过程的区别，并解释为什么上述模型更好。<br/>
$\Delta S = \mu \Delta t + \sigma \epsilon \sqrt{\Delta t}$ <br/>
$\Delta S = \mu S \Delta t + \sigma \epsilon \sqrt{\Delta t}$ <br/>
$\Delta S = \mu \Delta t + \sigma S \epsilon \sqrt{\Delta t}$

股票价格的预期增长量和变化量都应该与股票当时价格成正比。因此以上过程都不如下式的描述准确

$$\frac{\Delta S}{S} = \mu \Delta t + \sigma \epsilon \sqrt{\Delta t}$$


## 12.6 伊藤引理

若变量 x 符合以下伊藤过程：

$$dx = a(x,t)~dt + b(x,t)~dz$$

那么关于 $x$ 和 $t$ 的可微函数 $G(x,t)$ 遵循以下伊藤过程：

$$dG = (\frac{\partial G}{\partial x}a + \frac{\partial G}{\partial t} + \frac{1}{2} \frac{\partial^2 G}{\partial x^2}b^2)~dt +  \frac{\partial G}{\partial x}b~dz$$

#### 证明

根据多元泰勒展开公式，对 $G(x,t)$ 作二阶展开，得：

$$\Delta G = \frac{\partial G}{\partial x}\Delta x + \frac{\partial G}{\partial t}\Delta t + \frac{1}{2!}(\frac{\partial^2 G}{\partial x^2}\Delta x^2 + 2\frac{\partial^2 G}{\partial x \partial t}\Delta x \Delta t + \frac{\partial^2 G}{\partial x^2}\Delta t^2)$$

由于 $\Delta x$ 也是 $\Delta t$ 的函数：

$$\Delta x = a(x,t)~\Delta t + b(x,t) \epsilon \sqrt{\Delta t}$$

忽略 $\Delta t$ 的高阶无穷小 $o(\Delta t)$，可得：

$$\Delta x^2 = b^2(x,t) \epsilon^2 \Delta t $$

由于 $\epsilon$ 服从 $N(0,1)$，可知 $\epsilon^2$ 服从 $\chi^2_1$，其中 $\chi^2_n$ 是卡方分布，满足 $E(\chi^2_n) = n$，$D(\chi^2_n) = 2n$。


因此，对于 $\epsilon^2 \Delta t$，它的期望值是 $\Delta t$，方差是 $2\Delta t^2$。

就整个 $\Delta G$ 来看：

$$\Delta G = \frac{\partial G}{\partial x}\Delta x + \frac{\partial G}{\partial t}\Delta t + \frac{1}{2}\frac{\partial^2 G}{\partial x^2}\Delta x^2$$

由于 $\Delta x$ 部分还有方差为 $\Delta t$ 扰动，因此方差为 $\Delta t^2$ 的扰动从数量级上就可以忽略。我们因此可以直接把 $\epsilon^2$ 近似为常数，也就是它的期望值 1。得到：

$$\Delta G = \frac{ \partial G}{ \partial x}(a\Delta t + b \epsilon \sqrt{\Delta t}) + \frac{\partial G}{\partial t}\Delta t + \frac{1}{2}\frac{\partial^2 G}{\partial x^2}b^2\Delta t$$

$$\Delta G = (\frac{\partial G}{\partial x}a +  \frac{\partial G}{\partial t} + \frac{1}{2}\frac{\partial^2 G}{\partial x^2}b^2)\Delta t + \frac{\partial G}{\partial x} b \epsilon \sqrt{\Delta t}$$

也写作：

$$ dG = (\frac{\partial G}{\partial x}a +  \frac{\partial G}{\partial t} + \frac{1}{2}\frac{\partial^2 G}{\partial x^2}b^2)~dt + \frac{\partial G}{\partial x} b~dz $$


#### 应用：远期合约

对于远期合约，假设到期时间为 $T$，无风险利率为 $r$，当前时间为 $t$，当前现货价格为 $S$，则远期合约的执行价格应该为$S$ 和 $t$ 的函数（否则存在套利机会）：

$$F = Se^{r(T-t)}$$

如我们在 12.3 介绍的，假设 $S$ 满足伊藤过程：

$$dS = \mu S dt + \sigma S dz $$

其中期望收益为 $\mu$，波动率为 $\sigma$。则 $F$ 的价格变化过程可以利用伊藤引理确定。

$\frac{ \partial F}{\partial S} = e^{r(T-t)}$，$\frac{\partial^2 F}{\partial S^2} = 0$，$\frac{\partial F}{\partial t} = -r S e^{r(T-t)}$，带入伊藤引理有：

$$dF = (a - r S) e^{r(T-t)}~dt + e^{r(T-t)}b~dz $$

而，$a$ 为期望收益 $\mu S$，$b$ 为波动率 $\sigma S$，带入有：

$$dF = (\mu - r ) F~dt + \sigma F~dz$$

#### 例1
> 假设 $G(S, t)$ 是关于股票价格 $S$ 和时间 $t$ 的函数，$\sigma_S$ 及 $\sigma_G$ 分别为 $S$ 和 $G$ 的波动率，证明当 $S$ 的期望回报上升 $\lambda \sigma_S$ 时，$G$ 的期望回报上升 $\lambda \sigma_G$，其中 $\lambda$ 是常数。

假设 $S$ 符合伊藤过程：

$$ dS = a~dt + b~dz$$

由伊藤引理可知：

$$ dG = (\frac{\partial G}{\partial x}a +  \frac{\partial G}{\partial t} + \frac{1}{2}\frac{\partial^2 G}{\partial x^2}b^2)~dt + \frac{\partial G}{\partial x} b~dz $$

因此，波动率有如下关系：

$$ \sigma_G = \frac{ \partial G }{ \partial x } \sigma_S $$

若 $S$ 的期望收益 $a$ 上升了 $\lambda \sigma_S$，则显然有 $G$ 的期望收益上升了 $\frac{ \partial G }{\partial x } \lambda \sigma_S = \lambda \sigma_G$。

#### 例2

>假设股票价格 $S$ 服从几何布朗运动：<br/>
$dS = \mu S~dt + \sigma S~dz$
其中期望回报率是 $\mu$，波动率是 $\sigma$。证明 $S^n$ 也服从几何布朗运动。

同样是伊藤引理的应用。令 $G(S,t) = S^n$。则有：

$$dG = (\frac{\partial G}{\partial S} a + \frac{\partial G}{\partial t} + \frac{1}{2}\frac{\partial^2 G}{\partial S^2}b^2)~dt + \frac{\partial G}{\partial S} b~dz$$

带入 $a = \mu S$，$b = \sigma S$ 可得：

$$dG = (n\mu + \frac{ 1}{ 2} n(n-1)\sigma^2) G~dt + n \sigma G~dz$$

因此 $S^n$ 与 $S$ 具有同样的形式，也满足几何布朗运动。其期望的收益率为 $n\mu + \frac{1}{2} n(n-1)\sigma^2$，波动率为 $n \sigma$。
