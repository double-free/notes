# The Greek letters

Greeks 是为描述期权持仓风险引入的，他们各自代表一个维度的风险。

## 18.4 Delta

_delta_ ($\Delta$) 表示期权价格变动与标的物价格变动之间的比率。若假设 $c$ 为看涨期权价格，$S$ 为对应的股票价格，则有：

$$\Delta  = \frac{\partial c}{\partial S}$$

![delta的计算](https://upload-images.jianshu.io/upload_images/4482847-6a997c1bdd11d59f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

例如，某股票价格为 \$100，它的某个看涨期权价格为 \$10。一个投资者卖了 20 份看涨期权（每份 100 share，共 2000 share）。这个投资者的仓位可以通过购买 $0.6 \times 2000 = 1200$ share 股票对冲，使整个组合的 delta 为 0，即 delta 中性。

值得注意的是，由于期权的 delta 是一直变化的，上面例子中的 delta 中性只能保持非常短的一段时间。

假设过一段时间，该股票价格涨到了 \$110，看涨期权的 delta 随之上涨到 0.65。则我们需要额外再买入 $0.05 \times 2000 = 100$ share 来保持 delta 中性。这样持续依据最新的 delta 进行对冲的过程被称为动态对冲(_dynamic hedging_)。否则被称为静态对冲。

### 18.4.1 欧式期权的 Delta

对于一个欧式看涨期权，利用 B-S-M 公式，我们可以得到：

$$\Delta_{call} = \Phi(d_1)$$

对于欧式看跌期权，则有：

$$\Delta_{put} = \Delta_{call} - 1$$

#### 证明
由 B-S-M 公式稍作变形（计算 $t$ 时刻而非 0 时刻的价格），我们可以得到：

$$c = S \Phi(d_1) - Ke^{-r(T-t)}\Phi(d_2)$$

我们不能简单得到结果，因为 $d_1$ 和 $d_2$ 都是关于 $S$ 的函数：

$$d_1 = \frac{\ln(\frac{ S }{ K }) + (r + \frac{ 1}{ 2} \sigma^2)(T-t)} {\sigma \sqrt{T-t}}$$

$$d_2 = d_1 - \sigma \sqrt{T-t}$$

我们利用求导公式有：

$$\frac{ \partial c}{ \partial S} = \Phi(d_1) + S\Phi'(d_1)\frac{ \partial d_1}{ \partial S} - Ke^{-r(T-t)}\Phi'(d_2)\frac{ \partial d_2}{ \partial S}$$

其中 $\Phi'(x)$ 是正态分布的密度函数：

$$\Phi'(x) = \frac{1}{ \sqrt{2\pi}}e^{-\frac{ x^2}{2}}$$

我们很容易得到：

$$\frac{\partial d_1}{\partial S} = \frac{\partial d_2}{\partial S} = \frac{1}{S\sigma \sqrt{T-t}}$$

而利用 $d_1 = d_2 + \sigma \sqrt{T-t}$ 可以得到：

$$S\Phi'(d_1) = \frac{S}{\sqrt{2\pi}}e^{-\frac{(d_2 + \sigma \sqrt{T-t})^2}{2}} = Se^{-d_2\sigma\sqrt{T-t} ~-\frac{1}{2}\sigma^2(T-t)}\Phi'(d_2)$$

代入 $d_2$：

$$d_2 = \frac{\ln(\frac{S}{K}) + (r - \frac{1}{2}\sigma^2)(T-t)}{\sigma \sqrt{T-t}}$$

可以得到：

$$S\Phi'(d_1) = Se^{-\ln(\frac{S}{K}) - r (T-t)} \Phi'(d_2) = Ke^{-r(T-t)}\Phi'(d_2)$$

因此有：

$$\frac{\partial c}{\partial S} = \Delta = \Phi(d_1)$$

对于看跌期权，可以由 put-call-parity 得到。

典型的 $\Delta$ 与标的物价格的关系图如下：

![Delta and stock price](https://upload-images.jianshu.io/upload_images/4482847-0c2df9aed1022989.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看到 call 和 put 两者趋势一致，差距 1.0。

call 的 $\Delta$ 与到期时间的关系图如下：
![Delta and expiration date](https://upload-images.jianshu.io/upload_images/4482847-64b250160166335f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

即，离到期日越近，OTM 会越来越不值钱，Delta 趋近于 0。 ITM 的 Delta 则会趋近于 1。这符合直观。

有种说法是，将 Delta 理解为期权到期时 In The Money 的概率。这并不是很准确，我们在 [第十四章](https://www.jianshu.com/p/dacaa5620388) 例 5 中已经证明了，对于一个欧式看涨期权，到期时 In The Money 的概率是 $\Phi(d_2)$ 而不是 $\Delta = \Phi(d_1)$。准确的说它只是这个概率的一个 proxy，如 [Delta of Calls vs. Puts and Probability of Expiring In the Money](https://www.macroption.com/delta-calls-puts-probability-expiring-itm/) 所描述的。

### 18.4.3 远期合约 Delta
我们已经知道一个远期合约的价值为：

$$f = S_0 - Ke^{-rT}$$

其中 $K$ 是交割价格， $T$ 是距离到期日时间。因此若股票价格变化 $\Delta S$，则远期合约价格也变化 $\Delta S$。这说明它的 Delta 恒等于 1.0，与现货一致。

### 18.4.4 期货的 Delta
期货的合约价格为：

$$F = S_0e^{rT}$$

其中 $S_0$ 为当前现货价格，$T$ 为距离交割日时间。
因此一个期货的 Delta 就是 $e^{rT}$，在 $r > 0$ 时有 $\Delta > 1$。这看起来可能很奇怪，因为期货是每天 settle，因此几乎能立即拿到这个钱。

有时我们用期货合约来达成 Delta 中性。假设 $T$ 是距离期货到期日时间， $H_A$ 是用标的物资产对冲需要的的仓位，$H_F$ 是用期货对冲时需要的仓位。有：

$$H_F = e^{-rT}H_A$$

欧式期货期权的 Delta 通常被定义为与 __期货价格（而非即期价格）__ 相关的期权价格变化率，即 __远期（而非即期）__ 的 Delta。

#### 证明
已知期货价格 $F$ 满足：

$$F = S_0 e^{rT}$$

如果期货的到期日为 $T_1$，期权的到期日为 $T_2$，那么可以将 $d_1$ 表示为：

$$d_1 = \frac{\ln(\frac{Fe^{-rT_2}}{K}) + (r + \frac{1}{2}\sigma^2)T_1}{\sigma \sqrt{T_1}} = \frac{\ln(\frac{F}{K}) + r(T_1 - T_2) + \frac{1}{2}\sigma^2T_1}{\sigma \sqrt{T_1}} $$

假设 $T_1 = T_2 = T$那么这个期权关于 __现货__ 的 Delta 就可以写为：

$$\Delta = \Phi(d_1) = \Phi(\frac{\ln(\frac{F}{K}) + \frac{1}{2}\sigma^2T} {\sigma \sqrt{T}})$$

由于现货的 Delta 是 1，而期货的 Delta 是 $e^{rT}$，我们可以得出该期权关于 __期货__ 的 Delta：

$$\Delta_F = e^{-rT} \Delta $$


#### 例 1
> 假设某白银期货交割日在 9 个月以后，交割价格为 \$8 每盎司。同时某个白银的欧式看涨期权到期日在 8 个月以后，执行价格为 \$8 每盎司。无风险利率为 12% 每年，白银的波动率为 18% 每年。<br/>
则 1000 份该白银期货的欧式看涨期权空头的 Delta 是多少？

由于期货期权的 Delta 定义为与 __期货交割价格__ 相关的变化率，因此这里我们需要计算：

$$d_1 = \frac{\ln(\frac{F}{K}) + r(T_1 - T_2) + \frac{1}{2}\sigma^2 T_1}{\sigma \sqrt{T_1}} = 0.00544331 $$

然后可以继续得到

$$\Delta = e^{-rT}\Phi(d_1) = e^{-0.12*9/12} \times 0.5022 = 0.4590$$

因此1000份该期权空头的 Delta 为 -459.0，注意这个 Delta 是关于期货的 Delta。


#### 例 2
> 在例 1 中，为了对冲必需的白银期货初始仓位是多少？如果直接用白银现货对冲，初始仓位又该是多少？如果用还有 12 个月到期的白银期货呢？忽略白银的存放费用。

1. 若用 9 个月后到期的期货进行对冲，则应该用 459.0 盎司。
2. 若用白银现货，则需要把远期 Delta 转为即期的，即 502.2 盎司。
3. 若用 12 个月到期的，则以即期的乘以 $e^{-0.12}$，得 445.4 盎司。

## 18.5 Theta

Theta( $\Theta$) 用于描述 portfolio 价值随时间的变化率，即 time decay。可以表示为：

$$\Theta = \frac{\partial c}{\partial t}$$

对于 call 有：

$$\Theta_{call} = -\frac{ S_0 \Phi'(d_1)\sigma} { 2 \sqrt{T-t}} - rKe^{-r(T-t)}\Phi(d_2)$$

对于 put 有：

$$\Theta_{put} = \Theta_{call} + rKe^{-r(T-t)} $$

其中

$$\Phi'(x) =  \frac{ 1}{ \sqrt{2\pi} }e^{-\frac{ x^2 }{ 2}}$$

我们通过公式发现，对于 call， $\Theta$ __总是小于 0 的__。即随着到期日临近，期权越来越不值钱。

但是有一个例外就是对于 deep in the money 的 put，$\Theta$ 可能为正。直观上，假设某个股票价格已经跌到接近0，随着到期日临近，该股票上涨的概率越低，所以它的 put 具有更大的确定性赚钱。

关于这点，可以参考 [股票期权的性质](https://www.jianshu.com/p/1a753c2272db) 中的欧式看跌期权的上下限分析部分。在其中我们指出了对于 deep in the money 的 put，它的时间价值为负。而期权随着到期日临近，时间价值总是趋近于0。因此必然有一个渐渐增大的过程。

#### 证明

我们需要利用 B-S-M 公式得出期权价格相对于时间的导数。注意，这里不能直接对 $T$ 求导，因为它是到期时间，我们实际需要对 $t$ 求导。

先将 B-S-M 写成在 $t$ 时刻而非 $0$ 时刻的表达形式：

$$c = S_t\Phi(d_1) - Ke^{-r(T-t)}\Phi(d_2)$$

其中：

$$d_1 = \frac{ \ln(\frac{S_t}{ K}) + (r + \frac{ 1}{ 2}\sigma^2)(T-t)}{\sigma \sqrt{T-t}}$$

$$d_2 = d_1 - \sigma \sqrt{T-t}$$

对 $t$ 求导，有：

$$\frac{\partial c}{\partial t} = S_t\Phi'(d_1)\frac{\partial d_1}{\partial t}  - Ke^{-r(T-t)}\Phi'(d_2)\frac{\partial d_2}{\partial t} - rKe^{-r(T-t)}\Phi(d_2)$$

我们已经在证明 $\Delta = \Phi(d_1)$ 过程中得到：

$$S_t\Phi'(d_1) = Ke^{-r(T-t)}\Phi'(d_2)$$

因此我们可以得到：

$$\frac{\partial c}{\partial t} = S_t\Phi'(d_1)\frac{\partial(d_1 - d_2)}{\partial t} - rKe^{-r(T-t)}\Phi(d_2) = -\frac{\sigma S_t}{2\sqrt{T-t}}\Phi'(d_1) - rKe^{-r(T-t)}\Phi(d_2)$$

对于一个 put，B-S-M 公式为：

$$p = c + Ke^{-r(T-t)} - S_t$$

显然有：

$$\frac{\partial p}{\partial t} = \frac{\partial c}{\partial t} + rKe^{-r(T-t)}$$

典型的 call 的 $\Theta$ 与股票价格关系的曲线如下：
![Theta and stock price](https://upload-images.jianshu.io/upload_images/4482847-17ab3884c8509bc0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

当股票价格足够高时，$d_1$ 和 $d_2$ 趋近于无限大。此时$\Phi'(d_1) \to 0$，$\Phi(d_2) \to 1$。因此有看涨期权的 $\Theta \to -rKe^{-r(T-t)}$。

$\Theta$ 与距离到期日时间的关系如下：
![Theta and time to maturity](https://upload-images.jianshu.io/upload_images/4482847-4aa2b88f0b6aa3a6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看到，对于所有类型的期权，$\Theta$ 是小于 0 的。即随着到期日临近，期权价格趋近于降低。对于 ATM 的看涨期权，当临近到期日($t \to T$)时，它的 $\Theta$ 趋近于负无穷。

综合来看，__短期的 ATM 期权具有绝对值最大的负 Theta__。

#### 例

> 一个期权的 Theta 为 -0.1 （以年计算）含义是什么？如果交易员认为股票价格和波动率都不会变化，那么这个期权的仓位应该是多少？

1. 这表示每经过一个很短的时间 $\Delta t$，期权的价格会变化 $-0.1\Delta t$。
2. 如果股票价格和波动率都不会变化，则该期权价格会随着到期日临近越来越低。交易员应该选择卖出 Theta 为负，且绝对值较大的期权。而由于 __短期的 ATM 期权具有绝对值最大的负 Theta__，应该卖出短期的 ATM 期权。

## 18.6 Gamma

Gamma ($\Gamma$) 表示 portfolio 的 Delta 随标的物价格变动的变化率。
假设某 portfolio 价值为 $\Pi$，标的物价格为 $S$，则有：

$$\Delta = \frac{\partial \Pi}{\partial S}$$

$$ \Gamma = \frac{ \partial \Delta} {\partial S} = \frac{ \partial^2 \Pi}{\partial S^2} $$

如果 Gamma 绝对值较小，那 portfolio 的 Delta 变化也较慢，因此维持这个 portfolio Delta 中性较容易。否则，即使在很短的时间内不进行 Rebalance，都
有较高的风险。

对于一个 __Delta 中性的__ portfolio，我们有以下关系（这里的 $\Delta$ 表示全微分）：

$$\Delta \Pi = \Theta ~ \Delta t + \frac{1}{2}\Gamma \Delta S^2$$

这个结论可以利用 Taylor 展开，代入 $\frac{ \partial \Pi}{ \partial S} = 0$， 并忽略 $\Delta t$ 的高阶项得到。

下面的图描述了股票价格变化对于不同 Gamma 的 portfolio 的价值影响。

![Portfolio value variance and stock price variance](https://upload-images.jianshu.io/upload_images/4482847-457d3a5c0e783bba.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

例如，某个 Delta 中性的 portfolio 的 Gamma 是 -10000，该公式表明，如果标的物资产价格在很短的时间内变动 +2 或者 -2，即使它在价格变动之前是 Delta 中性的，这个 portfolio 会损失 20000。

### 18.6.1 Gamma 中性

刚才的例子表明，我们希望一个 portfolio 有较小的 Gamma 以降低风险。

由于标的物资产的 Delta 为常数 1，因此它的 Gamma 为 0，不能用于调节 portfolio 的 Gamma。因此，不同于 Delta Hedging，我们需要使用和标的物资产 __非线形相关的__ 衍生品，例如期权，来进行 Gamma 的调节。

假设一个 portfolio Delta 中性，但有 -3000 的 Gamma。同时，某个 call option 的 Delta 和 Gamma 分别为 0.62 和 1.50。则可以通过买入 $3000/1.5 = 2000$ 单位的 call 来使这个 portfolio Gamma 中性。

但是，这样又引入了额外的 $0.62 \times 2000 = 1240$ 的 Delta，因此我们还需要卖出 1240 个标的资产，使得该 portfolio 的 Delta 和 Gamma 均为中性。

### 18.6.2 Gamma 计算

可以简单通过求导得到：

$$\Gamma = \frac{ \partial \Delta }{ \partial S } = \Phi'(d_1) \frac{\partial d_1}{\partial S} = \frac{\Phi'(d_1)}{\sigma S_0 \sqrt{T}}$$

其中

$$\Phi'(x) = \frac{ 1 }{ \sqrt{2 \ pi} } e^{-\frac{ x^2 }{ 2 }}$$

可以看出，$\Gamma > 0$ __总是成立的__。可以回想 Delta 关于股票价格的曲线，无论对于 put 还是 call，随着股票价格上升，Delta 总是增加的。即 $\Gamma = \frac{ \partial \Delta} { \partial S}> 0$。

$\Gamma$ 与股票价格 $S$ 关系如下所示：
![Gamma and stock price](https://upload-images.jianshu.io/upload_images/4482847-b6000743e31dfaf1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


$\Gamma$ 与距离到期日时间的关系如下所示：
![Gamma and time to maturity](https://upload-images.jianshu.io/upload_images/4482847-f40b98af8cd47250.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出：
- Gamma 一定为非负数
- 对于 ATM 的期权，Gamma 最大
- 随着到期日临近，ATM 的期权 Gamma 单调增加，而 ITM 和 OTM 都是先增加后减少至 0

#### 例 1

> 某公司对由标的物是货币的看涨/看跌期权组成的 portfolio 进行了 Delta 对冲。哪种情况下 portfolio 盈利情况较好？<br/>
a. 该货币汇率基本稳定 <br/>
b. 该货币汇率剧烈波动

从 portfolio 价值上讲，根据公式（这里$\Delta$ 表示全微分）：

$$\Delta \Pi = \Theta ~\Delta t + \frac{1}{2}\Gamma \Delta S^2$$

我们可以发现对于正的 Gamma，当标的物价格变动较大时，portfolio 价值会变高（可以回忆图 18.8）。

同时，期权多头具有正的 Gamma，Gamma 绝对值小则对冲效果好。根据公式可知，$\sigma$ 越小 Gamma 越大，因此如果汇率剧烈波动则 Gamma 会较小，对冲也较容易。

#### 例 2

> 例 1 中，若 portfolio 是空头，哪种情况好？

期权空头具有负的 Gamma，标的物价格变动小则 portfolio 价值变高。

## 18.7 Delta, Theta, Gamma 之间的关系

我们之前已经推导过 B-S-M 微分方程：

$$\frac{\partial \Pi}{\partial t} + \frac{\partial \Pi}{\partial S}rS + \frac{1}{2} \frac{\partial^2 \Pi}{\partial S^2}\sigma^2 S^2 = r\Pi$$

根据我们对 Delta, Theta, Gamma 的定义，上式可以改写为：

$$\Theta + rS \Delta + \frac{1}{2} \sigma^2 S^2 \Gamma  = r \Pi $$

那么对于 Delta 中性的 portfolio，我们有：

$$\Theta + \frac{ 1 }{ 2 } \sigma^2 S^2 \Gamma  = r \ \Pi $$

我们发现，如果一个 Delta 中性的 portfolio 的 $\Theta$ 是一个很大的负数，则 $\Gamma$ 会是一个很大的正数，反之同理。

这也就是上一章中提到的， $\Theta$ 可以作 $\Gamma$ 的一个 “proxy”。

#### 例 1
> 计算 Delta, Theta, Gamma 的关系式，针对：
a. 货币的衍生品
b. 期货的衍生品



## 18.8 Vega

我们之前为了简化，一直把 volatility 作为一个常数。实际上，volatility 也是不断变化的。衍生品的价格同样会由于 volatility 的变化而变化。我们将这个变化率称为 Vega ($\nu$)：

$$\nu = \frac{ \partial \Pi}{ \partial \sigma}$$

对于欧式看涨和看跌期权，有：

$$\nu = \frac{ \partial c}{ \partial \sigma} = \frac{ \partial p}{ \partial \sigma}  = S_0 \sqrt{T} \Phi'(d_1)$$

证明较简单，直接求导并利用 $S_0\Phi'(d_1) = Ke^{-rT}\Phi'(d_2)$ 就可以得到。可以看出必然有 $\nu > 0$，这与我们之前的结论吻合，volatility 上升会引起看涨和看跌期权价格上升。

![Vega and stock price](https://upload-images.jianshu.io/upload_images/4482847-208930cc089db341.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

__标的物的 Vega 为 0__。因此我们如果需要 Vega 中性，需要通过买卖期权实现。一般而言，我们需要 2 种以上的期权的组合来实现 Gamma 和 Vega 同时呈中性。


#### 例 1
> 考虑一个 Delta 中性的 portfolio，它现在具有 -5000 的 Gamma 和 -8000 的 Vega。有两种可以交易的期权，如下表所示：

| | Delta | Gamma | Vega |
| :-: | :-: | :-: | :-: |
| Portfolio | 0 | -5000 | -8000 |
| Option 1| 0.6 | 0.5 | 2.0 |
| Option 2| 0.5 | 0.8 | 1.2 |

我们需要 Vega 和 Gamma 同时中性，假设两者数量配比分别为 $w_1$ 和 $w_2$，则：

$$0.5w_1 + 0.8w_2 = 5000$$

$$2.0w_1 + 1.2w_2 = 8000$$

可以解出 $w_1 = 400$，$w_2 = 6000$。买入 400 份 Option 1 和 6000 份 Option 2 后，Delta 从 0 变为 3240，所以还需要卖出 3240 份标的资产。

Gamma 中性可以在 Delta Hedging 的间隔时间里保护 portfolio 的价值不因为标的资产的价格变化产生变动，而 Vega 中性可以在 volatility 变化的时候保护 portfolio 的价值。需要根据对冲再平衡间隔时间以及波动率的波动率来决定选择哪些期权。


#### 例 2

> 什么情况下 2 个指数的欧式期权组成的 portfolio 可以同时 Gamma 和 Vega 中性？

即以下关于 $w_1$，$w_2$ 的方程组有非零解：

$$\Gamma_1 w_1 + \Gamma_2 w_2 = 0$$

$$\nu_1 w_1 + \nu_2 w_2 = 0$$

可以得到

$$\Gamma_1 \nu_2 = \Gamma_2 \nu_1$$

两个期权基于同一个标的物，因此必然有同样的 $S_0$, $q$, $r$, $\sigma$。唯一不同的就是到期日 $T$ 和执行价格 $K$，这也导致具有不同的 $d_1$ 和 $d_2$。

$$\Gamma_1 \nu_2 = \frac{1}{\sigma} e^{-q(T_1 + T_2)} \sqrt{\frac{T_2}{T_1}} \Phi'(d_{1(1)}) \Phi'(d_{1(2)}) $$

$$\Gamma_2 \nu_1 = \frac{1}{\sigma} e^{-q(T_1 + T_2)} \sqrt{\frac{T_1}{T_2}} \Phi'(d_{1(1)}) \Phi'(d_{1(2)}) $$

要两者相等，必然有 $T_1 = T_2$，即期限相同。


#### 例 3

> 某银行持有的美元/欧元汇率期权头寸有 30000 的 Delta 和 -80000 的 Gamma。<br/>
a. 解释这些数据的含义 <br/>
b. 假设目前汇率是 0.90 美元兑换 1.0 欧元，则如何使头寸 Delta 中性？<br/>
c. 在很短的时间后，欧元升值，汇率变为 0.93 美元兑换 1.0 欧元，则新的 Delta 为多少？如果要保持其 Delta 中性，需要什么额外的交易操作？<br/>
d. 假设该头寸最初是 Delta 中性的，那么它从汇率的变化中赚了还是赔了？

1. 说明若欧元升值 0.01 美元，则该头寸的价值上升 300，Delta 降低 800。该银行应该是持有欧元看跌期权的空头，因此具有正的 Delta 和负的 Gamma。
2. 需要卖空 30000 欧元。
3. 新的 Delta 为 30000 - 80000*0.03 = 27600，如果已经卖空 30000 欧元使其 Delta 中性，那么还需要买入 2400 欧元以保持 Delta 中性。
4. 由于最初 Delta 中性，Gamma 为负，因此欧元汇率在最初肯定大于 0.9 美元，在下降过程中，Delta 变为正。而它持有欧元看跌期权的空头，因此欧元汇率下跌，这个头寸有亏损。

## 18.9 Rho

Rho 描述了 portfolio 价值随利率 $r$ 的变化率：

$$\rho = \frac{\partial \Pi}{\partial r}$$

通过简单的求导可以得到：

$$\rho_{call} = KTe^{-rT}\Phi(d_2)$$

再利用 put-call-parity 可以得到：

$$\rho_{put} = -KTe^{-rT}\Phi(-d_2)$$

我们发现，利率上升会促使 call 的价值上升，put 的价值下降。这是符合我们直观的。当利率上升（假设资产价格不变），执行价格折现后变少。因此 call 相当于以更便宜的价格买到了资产，价格会升高。put 则反之。

由于利率不经常变化，这个 Greek 并没有其他几个受关注。


## 18.10 实际应用

对于一个 portfolio 的每个 Greek，交易公司都会设置一个上限。

Delta 的上限通常表示为标的物最大仓位。例如对于一个股票的 cash Delta 上限是 100 万，股票价格为 50，则最大仓位为 2 万。

Vega 的上限通常表示为 1% 的 volatility 变化引起的 portfolio 价值变化的上限。

期权交易员会尽量使得自己在每天交易结束时 Delta 中性。Gamma 和 Vega 也会监控，但是不会要求每天都是中性。

我们从 Gamma 和 Vega 的曲线可以看出，ATM 的期权具有最高的 Gamma 和Vega。随着股价变化，这些期权慢慢变为 OTM 或者 ITM，Gamma 和 Vega 就会自然减小。

## 18.12 公式的推广

我们目前导出的公式都是针对无股息股票的欧式期权。

假设股票的股息收益以复利计算为 $q$，则我们可以将 $S_0$ 替换为 $S_0 e^{-qT}$，得到适用于有收益的标的物的 B-S-M 公式：

$$c = S_0 e^{-qT} \Phi(d_1) - K e^{-rT} \Phi(d_2)$$

$$p = K e^{-rT} \Phi(-d_2) - S_0 e^{-qT} \Phi(-d_1)$$

$$d_1 = \frac{\ln(S_0 / K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}$$

$$d_2 = d_1 - \sigma \sqrt{T}$$

根据这个推导出其各个 Greeks 的计算公式：


|Greeks|European call|European put|
| :-: | :-: | :-: |
| Delta | $e^{-qT}\Phi(d_1)$ | $\Delta_{call} - e^{-qT}$ |
| Theta| $q e^{-qT} S_0 \Phi(d_1) - rKe^{-rT}\Phi(d_2) - \frac{e^{-qT}S_0 \Phi'(d_1)\sigma}{2\sqrt{T}}$ | $\Theta_{call} + rKe^{-rT}$ |
| Gamma| $\frac{ e^{-qT}\Phi'(d_1) }{ \sigma S_0\sqrt{T} }$ | $\Gamma_{call}$ |
|Vega | $e^{-qT}S_0\sqrt{T}\Phi'(d_1)$ | $\nu_{call}$ |
|Rho | $KTe^{-rT}\Phi(d_2)$ | $\rho_{call} - KTe^{-rT}$ |

1. 无股息的情况其实是上面的一个特例，令 $q = 0$ 即可得到。

2. 令 $q = r$，则可以得到期货期权的公式，__此时 Rho 不能使用通用公式__。$\rho_{call} = -cT$，$\rho_{put} = -pT$。这是由于 $q = r$，对 $r$ 求导时不能把 $q$ 作为常数。

3. 当我们分析货币期权时，有两个利率，本币无风险利率 $r$ 和外币无风险利率 $r_f$，令 $q = r_f$ 即得到对货币期权的公式。

#### 例 1
> 某个金融机构刚刚出售了 1000 份 7 个月后到期的标的物是日元的欧式看涨期权。假设即期汇率是 0.8 美分兑换 1 日元，执行价格是 0.81 美分。美国的无风险利率是 8%，日本的无风险利率是 5%。日元波动率是 15%。计算该仓位的 Delta, Gamma, Vega, Theta, Rho。

我们可以将 $q = r_f = 0.05$ 代入上面表格中的公式进行计算，将执行价格统一换算为美元。

然后计算 $d_1$：

$$d_1 = \frac{\ln(S_0 / K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}} = 0.1016$$

$$d_2 = d_1 - \sigma \sqrt{T} = -0.0130$$

$$\Phi(d_1) = 0.5405 $$

$$\Phi(d_2) = 0.4948 $$

$$\Phi'(d_1) = \frac{1}{\sqrt{2\pi}}e^{-\frac{d_1^2}{2}} = 0.3969$$

|Greeks|European call|Value|
| :-: | :-: | :-: |
| Delta | $e^{-qT}\Phi(d_1)$ | 0.5250 |
| Theta| $q e^{-qT} S_0 \Phi(d_1) - rKe^{-rT}\Phi(d_2) - \frac{e^{-qT}S_0 \Phi'(d_1)\sigma}{2\sqrt{T}}$ | 0.0004 |
| Gamma| $\frac{e^{-qT}\Phi'(d_1)}{\sigma S_0\sqrt{T}}$ |420.6051 |
|Vega | $e^{-qT}S_0\sqrt{T}\Phi'(d_1)$ | 0.0024 |
|Rho | $KTe^{-rT}\Phi(d_2)$ | 0.0022 |

#### 例 2

> 一个股指的远期合约和期货具有同样的 Delta 吗？解释你的结论。

股指期货会定期派发股息，假设其股息收益按复利计算为 $q$ ，则远期合约多头价值为：

$$S_0e^{-qT} - Ke^{-rT}$$

其 Delta 为 $e^{-qT}$。

对于期货，其约定的执行价格满足：

$$K = S_0e^{(r-q)T}$$

虽然它仅仅是未来的执行价格，但是由于期货每日结算制度，投资者能立即拿到收益。所以其即期价值就是执行价格，Delta 为 $e^{(r-q)T}$。

因此两者的 Delta 不同，期货 Delta 是远期 Delta 的 $e^{rT}$ 倍。

#### 例 3

> 期货的欧式看涨期权与期货价格 $F_0$ 关系式为：<br/>
$c = e^{-rT}[F_0 \Phi(d_1) - K \Phi(d_2)]$ <br/>
其中 <br/>
$d_1 = \frac{\ln(F_0 / K) + \frac{1}{2}\sigma^2 T}{\sigma \sqrt{T}}$ <br/>
$d_2 = d_1 - \sigma \sqrt{T}$ <br/>
a. 证明 $F_0 \Phi'(d_1) = K \Phi'(d_2)$ <br/>
b. 证明其 Delta 等于 $e^{-rT}\Phi(d_1)$ <br/>
c. 证明其 Vega 等于 $F_0 \sqrt{T} \Phi'(d_1) e^{-rT}$ <br/>
d. 证明其 Rho 等于 $-cT$

(a)
由于

$$\Phi'(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$

代入可知：

$$F_0\Phi'(d_1) = \frac{F_0}{\sqrt{2\pi}} e^{-\frac{d_2^2 + 2\sigma \sqrt{T}d_2 + \sigma^2 T}{2}}$$

而

$$2\sigma \sqrt{T}d_2 + \sigma^2 T = 2\ln(\frac{F_0} {K})$$

代入可以得到：

$$F_0\Phi'(d_1) = \frac{F_0}{\sqrt{2\pi}} \frac{K}{F_0} e^{-\frac{d_2^2} {2} } = K \Phi'(d_2)$$

(b)

$$\Delta = \frac{\partial c}{\partial F_0} = e^{-rT}[\Phi(d_1) + F_0 \Phi'(d_1) \frac{\partial d_1}{\partial F_0} - K \Phi'(d_2) \frac{\partial d_2}{\partial F_0} ]$$

利用 (a) 中的结论，有：

$$\Delta = e^{-rT}\Phi(d_1)$$

(c)

$$\nu = \frac{\partial c}{\partial \sigma} = e^{-rT}[F_0 \Phi'(d_1)\frac{\partial d_1}{\partial \sigma} - K \Phi'(d_2) \frac{\partial d_2}{\partial \sigma}]$$

利用 (a) 中的结论，有：

$$\nu = e^{-rT} F_0 \sqrt{T} \Phi'(d_1)$$

(d)

$$\rho =  \frac{\partial c}{\partial r} = -Te^{-rT}[F_0 \Phi(d_1) - K \Phi(d_2)] = -cT$$


## Appendix: Greeks from Trader's perspective

以上都是以工科思维来考虑 Greeks。那么在实际交易中，我们怎么利用 Greeks 呢？

在实际交易中，我们往往交易的都是近月的期权和期货。几乎可以忽略贴现问题。即，期货价格和现货价格我们认为是几乎相等的。在以下的内容中，我们都不再涉及贴现问题，也不刻意区分现货和期货价格。

### A.1 cash Greeks
在实际交易中，我们常常需要比较不同 underlying，不同 portfolio 的风险。因此我们引入了 cash Delta, cash Gamma 等概念。实际上就是将 Greeks 转为实际的现金。

假设某标的资产的价格为 $S$。它的 1 份期货合约包括标的资产为 $M$ 份(multiplier)。则对于该期货以及其期权衍生品，我们可以得出以下关系：

| cash Greek | 定义 | 计算 |
| :-: | :-: | :-: |
| cash Delta | 由 Delta 决定的 portfolio 的价值 | $S \times M \times \Delta$ |
| cash Gamma | 当 underlying 价格变化 1% 时 cache Delta 的变化量 | $\frac{S^2}{100} \times M \times \Gamma$ |

主要说下 cash Gamma 的计算。由于 $\Gamma$ 定义为标的资产价格变化 1 时 $\Delta$ 的变化量，因此 $\frac{S}{100} \Gamma$ 就是标的资产价格变化 1% 时 $\Delta$ 的变化量。再乘以系数 $S \times M$ 就是对 cash Delta 的影响。

> 玉米价格为 $3.5 每 bushel，一个期货合约交易 5000 bushels。假设我们持有 3 个 Delta 为 0.3 的看涨期权，那 cashe Delta 是多少？

$$3 \times 0.3 \times 3.5 \times 5000 = 15750$$

即我们 long 15.75k cash Delta。若玉米价格上升 1% 将盈利 $157.5，反之亏损 $157.5。

> 原油价格为 $50 一桶，一个期货合约交易 1000 桶原油。某原油为标的物的 portfolio 如果价格上升 1% 会导致 +3 Delta，cash Gamma 是多少？

将 3 Delta 转化为 cash Delta：

$$3 * 50 * 1000 = 150000$$

也就是说 1% 的原油价格变化会导致我们增加 150k 的 cash Delta。因此我们 long 150k cash Gamma。

> 原油现货价格为 \$104.8，期货的 multiplier 为 1000。已知执行价格为 \$105 的看涨期权 Gamma 为 0.1527。现在持有 50 个 \$105 的 straddle。
(a) Gamma 为多少？cash Gamma 为多少？
(b) 假设原油价格上涨 0.5%，持仓的 Delta 如何变化

(a)
straddle 是买入同样执行价格的 call 和 put，他们具有同样的 Gamma。

$$\Gamma = 0.1527 \times 2 \times 50 = 15.27$$

$$cash~\Gamma = \frac{104.8^2}{100} \times 1000 \times \Gamma = 1.677 \times 10^6$$

(b)
正的 Gamma 导致的 Delta 变化为：

$$0.5\% \times 104.8 \times \Gamma = 8 $$

即额外 long 8 Delta。

### A.2 Gamma 和 Theta

当我们持有期权时，Gamma 为正。此时，若标的物价格上涨，Delta 也随之上涨。我们需要卖出额外的标的资产。相反，当标的物价格下跌，Delta 也随之下跌，我们需要买入额外的标的资产。
如果不考虑 Theta，以上的交易看起来似乎有利可图。期权带来的Gamma 鼓励我们低买高卖得到收益，那为什么需要卖出期权呢？事实上，持有期权时 Theta 一般会带来损失。

在 18.6 中我们已经证明，于一个 __Delta 中性的__ portfolio，我们有以下关系（这里的 $\Delta$ 表示全微分）：

$$\Delta \Pi = \Theta ~ \Delta t + \frac{1}{2}\Gamma \Delta S^2$$

上式说明，在保持 Delta 中性时，portfolio 价值变化可由 Theta 和 Gamma 共同决定。我们需要比较到底是在 Theta 上亏的钱多还是在 Gamma 上赚的钱多。

我们可以令 $\Delta t$ 为一天，此时 $\Delta S$ 的最好估计就是将 $\sigma$ 也转为一天的变化后再乘以 $S$。当以下关系成立时，我们认为仓位是 "effecient" 的：

$$  \Theta t_{day} + \frac{1}{2} \Gamma  \sigma_{day}^2 S^2 > 0$$

而

$$\Gamma_{cash} = \frac{S^2} {100} \times M \times \Gamma $$

我们可以得出，每天持仓的现金 Decay 应该小于：

$$ \frac{1}{2} \times 100 \times \sigma_{day}^2 \Gamma_{cash} $$

> 假设我们仓位的 cash Gamma 为 1,000,000，波动率为 16%。问我们能接受的最大的 Decay 是多少？

一年有约 250 个交易日，由波动率为 16%，可以得出每天的波动率大约在 1%。可以计算能接受的最大 Decay：

$$ 0.5 \times 100 \times 0.01^2 *10^6 = 5000$$

### A.3 Slippage

slippage 定义为通过交易标的物来对冲 Delta 的成本。对于期权交易者来说，由于必须维持中性的 Delta，这是必然的成本。理想情况下我们能在理论价格上去交易，实际上，我们有时候不能完全在 top of book 成交，有时甚至需要交易几个 level 来对冲。这将影响我们在期权交易中的收益。

举例来讲，这是某个时刻原油期货的 book（仅列出 3 个 level）。

|Bid Quantity| Price| Ask Quantity|
| :-: | :-: |  :-: |
| | 66.87 | 31 |
| | 66.86 | 59 |
| | 66.85 | 38 |
| 30 | 66.84 | |
| 59 | 66.83 | |
| 50 | 66.82 | |

此时的理论价格是 66.843 （该价格通过 book 得出，具体不展开）

同时，与该原油期货相同到期日的两个正在交易的期权：

| # | Delta | Bid Price | Theo | Ask Price |
| :-: | :-: | :-: |  :-: | :-: |
| 1 | 0.30 | 1.23 | 1.237 | 1.24 |
| 2 | 0.15 | 0.63 | 0.640 | 0.65 |

以下实际例子可以帮助理解对冲成本（slippage），我们同时还可以引入一个 Retained Edge 概念。这是用于表示在 hedge 之后我们还能保留的来自期权交易的 edge。

| Operation | Option Edge | Hedge Loss | Retained Edge |
| :-: | :-: |  :-: | :-: |
| buy option #1| 1.237 - 1.230 = 0.007 | 0.003*0.3 = 0.0009 | 0.0061 (87%)|
| buy option #2| 0.640 - 0.63 = 0.01 | 0.003*0.15 = 0.00045 | 0.00955 (96%)|
| sell option #1| 1.24 - 1.237 = 0.003 | 0.007*0.3 = 0.0021 | 0.0009 (30%)|
| sell option #2| 0.65 - 0.64 = 0.01 | 0.007*0.15 = 0.00105 | 0.00895 (90%)|

或者，也可以从另一种方式理解。期权的理论价格由期货的 book 决定。我们在交易期权后，需要在期货上 hedge。这个行为会导致期货的 book 变动，从而导致期权的理论价格向 __不利__ 我们的方向更新。Retained Edge 是指利用更新后的理论价格得到的 edge，而非交易期权时刻的理论价格。

对于上面表格中的第一个例子，交易期权时理论价格是1.237，此后去 hedge 时，由于我们需要跨越 bid ask spread 去卖出期货，交易发生时刻期货的理论价格不再是 66.843，而是期货的 top of book bid，即66.84。

期货理论价格变动了 -0.003，乘以期权 #1 的 Delta 0.3，可以得出期权的理论价格变动为 -0.0009，新的期权价格为 1.2361。此时再来看这笔交易的 Edge，就是 Retained Edge，1.2361 - 1.23 = 0.0061。与表格中结论一致。

## 参考
1. [在线正态分布计算器](http://www.mwrf.net/tool/mathematics/2014/13618.html)

2.  Trading 101
