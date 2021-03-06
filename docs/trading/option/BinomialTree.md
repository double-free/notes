# Binomial Trees


_binomial tree_ 是 John Cox, Stephen Ross 和 Mark Robinstein 三位教授提出的一种不需要高级数学知识的期权定价方法。最初是用于教学的，后来广泛应用于美式期权的定价（因为可以随时执行，无法用 BSM 公式定价）。

这个方法构造了一个树状图表示在期权期限内可能会出现的股票价格变动的路径。在树的每一步（可能按时间来划分），股票价格都有一定的机会上行或者下行。当步长足够小时，股票价格趋于对数正态分布，也就是 Black-Scholes-Merton 模型的假设。因此，用二叉树估计的期权价格收敛于 Black-Scholes-Merton 模型的定价。

## 12.1 无套利假设下的一步二叉树模型

现在我们来推导一下期权的二叉树定价公式。

我们考虑如下 portfolio A：

1. 一定数量的股票
2. 一个 short call，在 T 时刻到期，执行价格为 K


我们将要用到的符号约定如下：

- $S_0$: 期权在 0 时刻价值
- $u$: 股票在 T 时刻上涨幅度，满足 $S_u = S_0u$
- $d$: 股票在 T 时刻下跌幅度，满足 $S_d = S_0d$
- $c$: 在 0 时刻，期权的价格，即我们需要求得的变量
- $f_u$: 在 T 时刻，股票价格上涨时的 payoff，对于欧式看涨期权是 $\max(S-K, 0)$
- $f_d$: 在 T 时刻，股票价格下跌时的 payoff，对于欧式看涨期权是 $\max(S-K, 0)$
- $\Delta$: 使得 portfolio A 无风险的股票仓位

所谓无风险，即无论股票价格上升还是下降，都不影响 portfolio A 的价值。

因此，我们可以得到：

$$\Delta S_0 u - f_u = \Delta S_0 d - f_d$$

注意，由于是看涨期权的空头，因此都是减去 payoff。
我们可以解出 $\Delta$：

$$\Delta = \frac{f_u - f_d}{S_0 u - S_0 d}$$

由于不存在套利机会，而这个 portfolio 是无风险的，因此它的收益必然等于无风险利率 $r$。否则，可以借钱买入该 portfolio，或者卖空来套利。

$$\Delta S_0 u - f_u = (\Delta S_0 - c)e^{rT}$$

即

$$c = (1 - ue^{-rT}) \Delta S_0  + f_u e^{-rT}$$

我们将 $\Delta S_0 =  \frac{f_u - f_d}{u - d}$ 带入，有

$$ c = \frac{(1 - de^{-rT})f_u - (1 - ue^{-rT})f_d}{u - d}$$

我们引入 $p = \frac{e^{rT} - d}{u - d}$，则可以将上式写为：

$$ c = e^{-rT}[p f_u + (1-p) f_d]$$

#### 例 1

我们可以用一个实际例子来说明如何使用这个公式。

假设目前的无风险利率是12%。一个股票现在价值为 \$20，我们已知 3 个月后它可能上涨到 \$22，也可能下跌到 \$18。则期限是 3 个月，执行价格是 \$21 的欧式看涨期权价格是多少？

我们可以得到 $u = 1.1$，$d = 0.9$，$r = 0.12$，$T = 0.25$，$f_u = 1$，$f_d = 0$。

于是计算得出 $p = 0.652$，call 的价格为$c = 0.633$。

## 12.2 风险中性定价

我们对风险的态度可以分为以下三种类型：

- 风险厌恶
  投资者期望从风险更高的投资中获取更高的收益。这也是现实世界中大部分投资者的态度。
- 风险中性
  投资者并不要求从风险更高的投资中获取更高的收益。
- 风险偏好
  投资者愿意从高风险的投资中获取更少的收益。

用抛硬币的赌局来举例，假设一枚均匀硬币，正面则投资者获益 100，背面则损失本金。因此该赌局期望获益 50 块。那么：

- 风险厌恶的投资者最多愿意花低于 50 块本金参与。
- 风险中性的投资者最多愿意花 50 块本金参与。
- 风险偏好的投资者最多愿意花高于 50 块本金参与。

我们常常假设一个风险中性的世界来简化衍生品定价：
1. 投资品的期望回报率等于无风险利率
2. 期权 payoff 的折现率等于无风险利率

我们再来看之前得到的期权定价公式：

$$ c = e^{-rT}[p f_u + (1-p) f_d]$$

我们来计算一下股票价格在 $T$ 时刻后的期望：

$$E(S_T) = p S_0 u + (1-p) S_0 d = p S_0 (u - d) + S_0 d$$

代入 $p = \frac{e^{rT} - d}{u - d}$ 得到：

$$E(S_T) = S_0 e^{rT}$$

我们发现，假设 $p$ 是股票价格上涨到 $S_u$ 的概率，$1-p$ 是股票价格下跌到 $S_d$ 的概率，则股票收益率的期望就是无风险利率。也就是说，股票这种存在风险的投资产品的期望收益率与无风险利率相等，这与 __风险中性__ 的世界相符。

## 12.2.1 风险中性假设下的一步二叉树模型

在 12.1 中，我们使用无套利的假设得到 $p = \frac{e^{rT} - d}{u - d}$。基于风险中性假设，我们可以使用另一种方法，即根据股票的期望价值来计算。

$$ p^{*} S_{u} + (1-p^{*}) S_{d} = 20 e^{rT}$$

带入公式也可以得到 $p^{*} = 0.652$。

对于 call 的价值，因为它在 $T$ 时刻有 $p$ 的几率价值 $S_u - K$，也有 $1-p$ 几率价值0，因此其当前价值为其 payoff 期望折现后的结果：

$$c = p (S_u - K) e^{-rT} = 0.633 $$

与 12.1 中的结果相同。即利用无套利假设和风险中性假设获得的结果相同。

### 12.2.2 Real World

需要指出的是，$p$ 是 __风险中性__ 假设下股票价格变为$S_u$ 的概率。一般情况下，这和现实中并不相等。因为现实世界是风险厌恶型的，因此对于高风险的股票会有更高的收益期望。

现实中当无风险收益率是 12% 时，假设对股票的收益期望是 16%，则

$$ 22 p^* + 18(1 - p^*) = 20e^{0.16*3/12}$$

计算得知 $p^* = 0.704$。

在 12.2.1 中我们计算期权价格时，我们采用了与股票相同的利率对 payoff 进行贴现，然而由于期权风险比股票大，我们对期权收益的贴现率应该大于 16%。我们无法知道实际的贴现率因此无法计算。

只有在风险中性假设下，我们对所有的资产回报率的期望值才能都等于无风险利率，从而可以计算期权的价格。

## 12.3 两步二叉树

![Stock prices in a two-step tree](https://upload-images.jianshu.io/upload_images/4482847-05ce26fb20fd61b1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

假设在上图中，每一步时间为 3 个月，股票可能上涨或者下跌 10%。

我们希望计算一个执行价格 21，期限 6 个月的 call 的价值。

由于每一步上涨/下跌幅度是一致的，因此计算所得的上涨/下跌的概率也是一致的。

我们直接从最终状态入手，

$$f = e^{-2r \Delta t}[p^2 f_{uu} + 2p(1-p)f_{ud} +(1-p)^2f_{dd}]$$

其中 $p = \frac{e^{rT} - d}{u - d}$。

### 12.3.1 Delta

我们之前在一步二叉树中已经用到了 $\Delta$，它是当我们卖出一份期权时，为了构造无风险组合需要持有的标的资产数量。它代表了期权价格变化与股票价格变化的比率。

因此，对于 12.1 例 1 中的 $\Delta$，我们用如下式子计算：

$$ \Delta = = \frac{f_u - f_d}{S_0 u - S_0 d} = \frac{1-0}{22-18} = 0.25$$

在两步二叉树中，我们发现 $\Delta$ 是随时间变化的。因此无法构造一个 portfolio 使它在到期日前都保持风险中性。在实际中，我们需要不断调节持有股票的数量来进行对冲。

## 12.5 美式期权

我们之前讨论的都是欧式期权，因此可以只依赖最终状态来确定价格。但是对于美式期权，由于可以提前行权，在每一步，期权的 payoff 都应该取以下两个值的最大值：

- 通过 $f = e^{-r \Delta t}[pf_u + (1-p)f_d]$ 计算出的 payoff
- 立即行权的 payoff

假设股票价格变化如下图。每步时间为 1 年，可能有 20% 的上涨或者下跌。无风险利率为 5%。
![Using a two-step tree to value an American put option](https://upload-images.jianshu.io/upload_images/4482847-ebf0926c29dd5d44.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

注意上图的 C 点。

我们首先计算 $p = \frac{e^{rT} - d}{u - d} = 0.6282$，则在 C 点时，$f = 9.4636$。然而，假设选择在 C 点时立即行权，我们可以获得 $12。因此在 C 点期权的价值为 12 而非 9.4636。

在 B 点，期权价值为 1.4147，在 A 点，期权价值为 5.0894。

## 12.7 使 $u$ 和 $d$ 与波动率吻合

在实际中，当我们构造二叉树来表示股票价格变化时，我们需要选取恰当的 $u$ 和 $d$ 使股票的变化和波动率 $\sigma$ 一致。那我们究竟需要匹配现实中的波动率，还是风险中性的波动率呢？

我们马上将要证明，__现实世界中的波动率和风险中性世界中的波动率是相等的__。

![Stock price changes in real world and risk neutral world](https://upload-images.jianshu.io/upload_images/4482847-d427171e099cc74a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

股票价格变化如上图所示，假设股票的波动率 $\sigma$，则股票回报率的方差为 $\sigma^2 \Delta t$。

对于现实世界，股票有 $p^*$ 的概率回报率为 $u -1$，同时有 $1-p^*$ 的概率回报率为 $d - 1$ 。由于 +1 并不影响方差，我们可以转为计算股票 __终值/初值__ 的方差。

$$D(x) = E(x^2) - [E(x)]^2$$

因此有

$$ [p^* u^2 + (1-p^*)d^2] - [p^*u + (1-p^*)d]^2 = \sigma^2 \Delta t$$

化简得到

$$ p^* (1-p^*)(u-d)^2 = \sigma^2 \Delta t$$


另一方面，假设现实中我们对股票收益的期望回报率是 $\mu$，则有：

$$ p^{*} S_0 u+ (1-p^{*}) S_0 d = S_0 e^{\mu \Delta t} $$

$$p^* = \frac{e^{\mu \Delta t} - d}{u - d} $$

从而得到

$$(e^{\mu \Delta t} - d)(e^{\mu \Delta t} - u) + \sigma^2 \Delta t = 0$$

将上式 Taylor 展开并忽略 $\Delta t$ 的高次项，可以得到 $u, d$ 的一组解：

$$u = e^{\sigma \sqrt{\Delta t}}$$

$$d = e^{- \sigma \sqrt{\Delta t}}$$

（我好像没有办法通过泰勒级数得到同样的结果，暂且先留着）
