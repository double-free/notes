# Reading Note: Option Profit and Loss Attribution and Pricing

一篇论文阅读笔记：

> "Option Profit and Loss Attribution and Pricing: A New Framework", by Peter Carr and Liuren Wu.

个人认为，这篇文章专注期权定价和风险分析的 "locality"。这个 locality 体现在：

1. 不以期权的长期的 terminal payoff，而是以短期的 implied volatility 以及 underlying price 来定价。
2. 不用 underlying 级别的统一 volatility，而是用期权自己的 implied volatility

传统的方法专注于找到一个适用于所有 contracts 的 "__中心化__" 的参数(例如 BSM 中的 volatility)，而这篇文章的方法专注于一个 contract 自身的属性，是一种 “__去中心化__” 的方法。

原文中提到，这两种方法是相辅相成的：

> The two approaches do not directly compete, but rather complement each other. Dynamics specifications from the traditional approach can provide insights for formulating hypotheses on moment condition estimation, while empirically identified co-movement patterns in the moment conditions can provide guidance for the specification of reference dynamics.

"moment" 是全文中多次提到的词，本文的语境下可认为是期权价格对于某一影响因素的导数。例如原文提到：

> This pricing equation is not based on the full specification of the underlying security price dynamics, but rather on the first and second conditional moments of the security price and the option’s implied volatility movements at time t.

在这个新方法框架下，基于风险中性以及无套利假设，单个期权的价格是和它自身的风险对消的。比较多个期权的价格，必须先比较他们的风险暴露。新方法本质上是通过确定各个风险暴露的系数，再依据假设“持有的期权在 theta 上的损失等于其在其他 greeks 上的期望 PnL” 来决定价格。


## 使用新方法进行期权定价

我们考虑一个 European call option，将其持有到到期日会有一个 terminal payoff，在传统方法中，我们会利用一些假设来简化并计算这个 __payoff 的期望值__，再利用这个期望值确定期权现在的价值。

在新的方法框架中，我们专注于 __瞬时的 PnL__ 而非最终 payoff。期权的短期 PnL 取决于各个 greeks 上的风险暴露，新方法是通过量化其风险的大小来定价，而非通过 payoff。

期权的瞬时 PnL 的影响因素有：

1. 距离到期日的时间 (time to maturity)
2. 标的物价格 (underlying price)
3. 隐含波动率 (implied volatility)

我们可以把 European call option 的价值近似表示为一个函数（__假设利率为 0.0__）：

$$ B(t, S_t, I_t; K, T) $$

注意，这个函数并不是基于 BSM 公式推导，而仅仅是确定了影响期权价格的自变量。更进一步，我们利用全微分公式，把期权价格的变化率表示为：

$$ \mathop{\mathrm{d} B} = [B_t \mathop{\mathrm{d} t} + B_S \mathop{\mathrm{d} S_t} + B_I \mathop{\mathrm{d} I_t}] + [\frac{1}{2} B_{SS} (\mathop{\mathrm{d} S_t})^2 + \frac{1}{2} B_{II} (\mathop{\mathrm{d} I_t})^2 + B_{IS} (\mathop{\mathrm{d} S_t} \mathop{\mathrm{d} I_t})] + J_t $$

其中，$d_X$ 表示变量 $X$ 极小的一个变化量。由于 underlying price 和 implied volatility 变动更加剧烈，是期权风险暴露的主要成分。因此，我们需要用二阶近似，而时间仅需要一阶近似。另外，$J_t$ 更高阶的扰动。各系数意义如下：

| Variable | Name | Description |
| :-: | :-: | :-: |
| $B$ | Option Price | |
| $B_t$ | Theta | time decay of option price in BSM formula |
| $B_S$ | Delta | `Delta` is first order greek, it is the ratio of change in option price to the change in underlying price. |
| $B_{SS}$ | Gamma | `Gamma` is second order greek, it is the ratio of change in `Delta` to the change in underlying price. |
| $B_I$ | Vega | `Vega` is first order greek, it is the ratio of change in option price to the change in volatility |
| $B_{II}$ | Volga | `Volga` is second order greek, it is the ratio of change in `Vega` to the change in volatility. |
| $B_{IS}$ | Vanna | `Vanna` is second order greek, it is the ratio of change in `Vega` to the change in underlying price. Alternatively, it is the ratio of change in `Delta` to the change in volatility |

截止目前还是正常的 BSM 公式的应用，我们已经把瞬时的盈亏转化到各个风险维度上，下一步则是得到 PnL 的期望值。

### 期权自身风险暴露与价格的关系

我们把刚才的公式用各变量 __变化率__ 表示，并对两边求期望，再除以瞬时 investment horizon，得到：

$$ \frac{E[\mathop{\mathrm{d} B}]}{\mathop{\mathrm{d}t}} = B_t + B_I I_t \mu_t + \frac{1}{2} B_{SS} S_t^2 \sigma_t^2 + \frac{1}{2}B_{II} I_t^2 w_t^2 + B_{IS} I_t S_t \gamma_t $$

其中：

$$\begin{aligned}
\mu_t &= E[\frac{\mathop{\mathrm{d} I_t}}{I_t}] / \mathop{\mathrm{d} t} \\
\sigma_t^2 &= E[(\frac{\mathop{\mathrm{d} S_t}}{S_t})^2] / \mathop{\mathrm{d} t} \\
\omega_t^2 &= E[(\frac{\mathop{\mathrm{d} I_t}}{I_t})^2] / \mathop{\mathrm{d} t} \\
\gamma_t &= E[(\frac{\mathop{\mathrm{d} S_t}}{S_t} \cdot \frac{\mathop{\mathrm{d} I_t}}{I_t})] / \mathop{\mathrm{d} t} \\
\end{aligned}$$

注意到上式中有 implied volatility 的期望变化率，却没有 stock return 的期望变化率。这是因为本文作者做了如下假设：

$$ E[\frac{\mathop{\mathrm{d} S_t}}{S_t}] = 0  $$

因为在 0 利率以及风险中性假设下，资产的 __期望回报率为 0__。同样由于这个假设，期权的投资回报率预期值也是 0，因此：

$$ 0 = \frac{E[\mathop{\mathrm{d} B}]}{\mathop{\mathrm{d}t}} = B_t + B_I I_t \mu_t + \frac{1}{2} B_{SS} S_t^2 \sigma_t^2 + \frac{1}{2}B_{II} I_t^2 w_t^2 + B_{IS} I_t S_t \gamma_t $$

这个式子揭示了短期期权交易的 trade-offs：

> By being long in an option, one loses time value as calendar time passes (captured by `Theta`). The loss is compensated by:
  1) underlying return variance $\sigma_t^2$, multiplied by `Gamma`,
  2) implied volatility variance $\omega_t^2$, multiplied by `Volga`,
  3) covariance of underlying return and implied volatility $\gamma_t$, multiplied by `Vanna`,
  4) implied volatility movement $\mu_t$, multiplied by `Vega`.

基于上述关系，我们判断一个 option 的价格是否合理就相当于判断这个 option 在 `Theta` 上的期望损失是否等于在 `Gamma`, `Vega`, `Volga`, `Vanna` exposures 上的期望盈利。

注意，如果我们假设 implied volatility 不变，则有:

$$ \mu_t = \omega_t^2 = \gamma_t = 0 $$

上式退化为 BSM 模型的情形 (BSM 也假设了波动率为常数)：

$$ B_t + \frac{1}{2} B_{SS} S_t^2 \sigma_t^2 = 0 $$

截止目前，我们并未用到 BSM 公式，也就是说上面的等式是对非 BSM 模型也成立的。接下来，我们探讨基于 BSM 模型假设下的各项系数关系。

假设 $t$ 是当前时间，$\tau$ 是距离期权到期的时间，由 BSM 公式，我们可以得到：

$$ B_{SS} = \mathrm{Gamma} = \frac{\Phi'(d_1)}{S_t I_t \sqrt{\tau}} $$

其中 $\Phi$ 是标准正态分布的 cdf。因此 cash Gamma 为：

$$ B_{SS} S_t^2 = \frac{S_t \Phi'(d_1)}{I_t \sqrt{\tau}} $$

同理得到 cash Theta 和 cash Vega，并以 cash Gamma 表示：

$$ B_t = \mathrm{Theta} = -\frac{1}{2} S_t I_t \Phi'(d_1) / \sqrt{\tau} = -\frac{1}{2} I_t^2 \cdot B_{SS} S_t^2 $$

$$ B_{I} I_t = S_t \sqrt{\tau} \Phi'(d_1) \cdot I_t = I_t^2 \tau \cdot B_{SS} S_t^2 $$

cash Volga:

$$\begin{aligned}
B_{II} I_t^2 &= \frac{\partial B_I}{\partial I_t} \cdot I_t^2 \\
             &= S_t \sqrt{\tau} \Phi''(d_1) \frac{\partial d_1}{\partial I_t} \cdot I_t^2 \\
             &= S_t \sqrt{\tau} \Phi'(d_1) d_1 [ \frac{\ln{S/K}}{I_t^2 \sqrt{\tau}} - \frac{1}{2} \sqrt{\tau}] \cdot I_t^2 \\
             &= S_t \sqrt{\tau} \Phi'(d_1) d_1 d_2 \frac{1}{I_t} \cdot I_t^2 \\
             &= d_1 d_2 I_t^2 \tau \cdot B_{SS} S_t^2
\end{aligned}$$

cash Vanna:

$$\begin{aligned}
B_{IS} I_tS_t &= \frac{\partial B_I}{\partial S_t} \cdot I_tS_t \\
             &= \sqrt{\tau} [\Phi'(d_1) + S_t \Phi''(d_1) \frac{\partial d_1}{\partial S_t}] \cdot I_t S_t \\
             &= \sqrt{\tau} \Phi'(d_1) (1 - \frac{d_1}{I_t \sqrt{\tau}}) \cdot I_t S_t \\
             &= -\Phi'(d_1) \frac{d_2}{I_t}  \cdot I_t S_t \\
             &= - d_2 I_t \sqrt{\tau}  \cdot B_{SS} S_t^2
\end{aligned}$$

将以上 cash greeks 代入公式，消去 cash Gamma，得到：

$$ I_t^2 = 2\mu_t I_t^2 \tau + \sigma_t^2 + \omega_t^2 d_1 d_2 I_t^2 \tau - 2 \gamma_t d_2 I_t \sqrt{\tau} $$

传统的期权定价把 __cross-sectional consistency 作为其首要目的__，为了达到这个目的，使用者需要建模标的资产的价格变动，基于这个统一的假设模型，算出期权 terminal payoff 的期望，再折现进行定价。由于使用了统一的标的资产价格模型，所有的期权定价拥有了一致性。

然而，上式则是一个更加局部的方法，它描述了 __期权自身风险和回报的 trade-off__。我们可以将期权的价值与它 underlying price 和 implied volatility 上的一阶、二阶 greeks 上的风险做比较。

### 通过历史数据确定系数

我们可以通过历史数据拟合获得 $\mu_t$, $omega_t^2$, $\gamma_t$。而 $\sigma_t^2$ 可以直接通过 underlying price 获取。

本章一共利用了 3 个假设，分别是：

1. expiry 接近的 at-the-money option $\mu_t$ 相同
2. 同一 expiry，在一定范围内不同 strikes 的 $\mu_t I_t^2$ 相同
3. 同一 expiry，在一定范围内不同 strikes 的 $\omega_t^2$ 和 $\gamma_t$ 相同

这三个假设可靠性后面会用实际数据验证。直观上与我们调节 volatility curve 的思路是类似的。

#### The At-the-Money Implied Variance Term Structure

为了将 term structure 和 moneyness effect 分开分析，我们定义 at-the-money 期权满足以下条件：

$$ d_2 = \frac{\ln(S_t/K) + \frac{1}{2} I_t^2 \tau}{I_t \sqrt{\tau}} = 0 $$

在上述条件下，期权的 `Volga` `Vanna` 都为 0，此外，由于 $d_2 = 0$ 等式可以简化为：

$$ I_t^2 = 2\mu_t I_t^2 \tau + \sigma_t^2 $$

> 注意，这里的 at-the-money 期权的定义并不是传统的 `Delta = 0.5` 的期权，`Delta = 0.5` 的期权是满足 $d_1 = 0$ 的，而这里是 $d_2 = 0$ 的。其中，$\Phi(d_2)$ 的现实意义是在到期时变为 in-the-money 的概率。

这里给我们的启示是，在学习中不要教条主义，moneyness 的定义完全是交易者根据自己分析需要定义的，at-the-money 的期权可以是 `S = K`（这样的期权没有intrinsic value，只有time value），可以是 `d1 = 0`（此时 `Delta = 0.5`），也可以是 `d2 = 0`（本文的定义，此时 Volga 和 Vanna 都为 0）。

我们进一步 __假设__ 两个到期日接近的 atm 期权的 implied volatility 变化率相近，即：

$$ \mu_t(\tau_1) \approx \mu_t(\tau_2) $$

则我们可以通过以下方式算出 $\mu_t$：

$$ \mu_t = \frac{A_t^2(\tau_2) - A_t^2(\tau_1)}{2(A_t^2(\tau_2) \tau_2 - A_t^2(\tau_1) \tau_1)} $$

其中 $A_t(\tau)$ 表示 $\tau$ 时间过后到期的 atm 期权的 implied volatility。

#### The Implied Volatility Smile

接下来，我们分析同一个 expiry 下，其他期权与 atm 期权的 implied volatility 关系。

我们把 $A_t(\tau)$ 作为已经计算出来的数值，然后用它来表示同一个 expiry 下其他期权的 implied volatility。可以得到：

$$ I_t^2 - A_t^2 = 2 \tau (\mu_t I_t^2 - \mu_t^A A_t^2) + \omega_t^2 d_1 d_2 I_t^2 \tau - 2 \gamma_t d_2 I_t \sqrt{\tau} $$

如果我们更进一步 __假设__ implied volatility 变化率与其平方成反比，即:

$$ \mu_t I_t^2 = \mu_t^A A_t^2 $$

> TOOD: is this a valid assumption?

则可以简化为：

$$ I_t^2 - A_t^2 = \omega_t^2 d_1 d_2 I_t^2 \tau - 2 \gamma_t d_2 I_t \sqrt{\tau} $$

即，对于每个 strike，它的 implied volatility 和 atm 期权的平方差可以由 implied volatility variance 和 volatility-return covariance 确定。因此，__波动率微笑曲线的形状是由它下一个时间片波动率可能的变化，以及波动率对 underlying price 变化的响应决定的__。

由于 implied volatility 和 d1, d2 可以直接结算得到，我们通过历史数据可以拟合出 $\gamma_t$ 和 $\omega_t^2$ 的值。

为了统一不同的 strikes，我们再次假设同一 expiry 下，在 at-the-money 附近范围内的 implied volatility 变化特性相同，即：

$$ \omega_t^2(k) = \omega_t^2, \gamma_t(k) = \gamma_t $$

由于 near-the-money 期权交易最活跃，我们可以用 atm 期权以及它附近的（即 $-\epsilon < d_2 < \epsilon$）的期权来计算 $\omega_t^2$ 和 $\gamma_t$。
