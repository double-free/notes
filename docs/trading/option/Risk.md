# Option Volatility and Pricing - Risks


"Option Volatility and Pricing", Sheldon Natenberg.

This book helps readers to understand option from a trader's perspective. Other relevant reading materials:

- Option, Future and other Derivatives (John Hull)
  - Best book to start with
- Stochastic Volatility Modeling (Lorenzo Bergomi)
  - Advanced book if you are a developer or quant.



Risk related Chapters:

- Chapter 7: Risk Measurement I
- Chapter 9: Risk Measurement II
- Chapter 10: Introduction to Spreading
- Chapter 13: Risk Consideration

## Chapter 7: Risk Measurement I


### 7.1 Difference between Gamma and Vega

Gamma is a measurement of _magnitude risk_. Options have positive gamma.

> A negative gamma position is a good indication that a trader either wants the underlying market to sit still or move only very slowly. A positive gamma position indicates a desire for very large and swift moves in the underlying market.

This seems to correspond to volatility. If we have a negative gamma, we want the market to remain relatively quiet. Isn't this the same as saying we want lower volatility?

>  The gamma is a measure of whether we want higher or lower **realized volatility** (whether we want the underlying contract to be more volatile or less volatile). The vega is a measure of whether we want higher or lower **implied volatility**.

Although the volatility of the underlying contract and changes in implied volatility are often correlated, this is not always the case.

> As the volatility of the underlying contract (realized volatility) changes, option demand rises and falls, and this demand is reflected in a corresponding rise or fall in the implied volatility.

## Chapter 9: Risk Measurement II

### 9.1 Implied delta

> many traders use the implied delta, the delta that results from using the implied volatility.

Because the delta depends on the volatility, but volatility is an unknown factor, calculation of the delta can pose a major problem for a trader, especially for a large option position.

### 9.2 Theta

> the rate of decay slows for in-the-money and out-of-the-money options, whereas it accelerates for an at-the-money option

![Theta](https://upload-images.jianshu.io/upload_images/4482847-04a759c482100230.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



This is because both OTM and ITM options have very little time value.

### 9.3 Vega

> Because vega is not a Greek letter, a common alternative in academic literature, where Greek letters are preferred, is kappa ($\kappa$).

The option price as volatility change is quite similar to theta one.
> In many situations, time and volatility will have a similar effect on options. More time, like higher volatility, increases the likelihood of large price changes. Less time, like lower volatility, reduces the likelihood of large price changes.

![vega](https://upload-images.jianshu.io/upload_images/4482847-60517e85dc868ab6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> vega of an at-the-money option is relatively constant with respect to changes in volatility. (the vega of an at-the-money option declines very slightly as we raise volatility. )

$$\nu = \frac{ \partial c}{ \partial \sigma} = \frac{ \partial p}{ \partial \sigma}  = S_0 \sqrt{T} \Phi'(d_1)$$

where:

$$d_1 = \frac{\ln(S/K) + (r + 0.5\sigma^2)T}{\sigma \sqrt{T}}$$

If the option is ATM, we have $S = K$, and if the interest rate is negligible (r = 0), we have:

$$d_1 = 0.5 \sigma \sqrt{T}$$

The sensitivity of vega to a change in volatility (volga):

$$\begin{align} \frac{ \partial^2 c}{ \partial \sigma^2} = \frac{ \partial^2 p}{ \partial \sigma^2} &= S_0 \sqrt{T} \Phi''(d_1) \frac{\partial d_1}{\partial \sigma} \\
&= -\frac{1}{4\sqrt{2 \pi}} e^{- \frac{\sigma^2 T}{8}} S_0 \sigma T^{\frac{3}{2}}
\end{align}$$

So it is negative, which means it is true that "the vega of an at-the-money option declines as we raise volatility".

As for the magnitude of change, the book says it is "very slight", let's verify it with formula (if ATM):

$$\frac{\text{volga}}{\text{vega}} = \frac{d_1 d_2}{\sigma} =  - \frac{1}{4} \sigma T$$

It seems negligible if the time to expire is small enough. let's verify with an ATM option with:

1. stock price $S=100$
2. expire in one month $T=1/12$
3. anual volatility $\sigma = 25\%$

Its vega is 11.51, usually we use 1% change in volatility so it becomes 0.1151, which means +1% volatility change will +0.1151 to ATM option price.

Its volga is 0.06, compared to 11.51 this is negligible.

### 9.4 Gamma

$$\Gamma = \frac{ \partial \Delta }{ \partial S } = \Phi'(d_1) \frac{\partial d_1}{\partial S} = \frac{\Phi'(d_1)}{\sigma S_0 \sqrt{T}}$$

> Gamma, Theta and Vega are greatest when an option is at the money.


![Gamma](https://upload-images.jianshu.io/upload_images/4482847-60173ae60c2f82b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

With less time to expiration or lower volatility, OTM and ITM options' time value will be lower because they will be less likely that a large price change will happen. Gamma will also be lower, the graph becomes "thin".

More time to expiration, like higher volatility, will make OTM and ITM options behave like ATM ones. The graph becomes "fat".


## Chapter 10: Introduction to Spreading

Scalping tries to make a market and make profit from liquidity.

> By observing the activity in a particular market, a scalper would try to determine an equilibrium price that reflected a balance between buyers and sellers. The scalper would then quote a bid-ask spread around this equilibrium price, attempting to buy at the bid price and sell at the offer price as often as possible without taking either a long or short position for any extended period of time.

However, option markets are rarely sufficient liquid to support this type of trading.

> Most successful option traders are spread traders. A **spread** is a strategy that involves taking opposing positions in different but related instruments.

When market condition changes, two legs of spread will gain and lose value respectively. Many common spreading strategies are based on __arbitrage__ relationships.

- cash-and-carry strategy
  Given the current cash price, interest rate, and storage and insurance costs, a commodity trader can calculate the value of a forward contract. If the actual market price of the forward contract is higher than the calculated value, the trader will create a spread by purchasing the commodity, selling the overpriced forward contract, and carrying the position to maturity.

Another type of spreading strategy involves buying and selling futures contracts of different maturities on the same underlying commodity. It is similar to the cash-and-carry strategy. Besides, we can also do this in different markets (intramarket spread v.s. intermarket spread).

cash-and-carry strategy cancels the the directional risk, the trader does not need to tell which contract is mispriced, he only need to know if the spread is profitable.

Note that the spread is not NECESSARILY a contract. We can't guarantee the entire spread will be executed at one time. Trade has to choose the best time to execute each leg. In this procedure, the portfolio is at risk. __Most traders learn that it is usually best to execute the more difficult leg first__.

### 10.1 Option Spreads

Spreading strategies are widely employed in option markets, because:

1. **A trader might perceive a relative mispricing between contracts**. Though it may not be possible to determine the exact value of either contract, the trade might be able to estimate the relative value of contracts.

> In option markets, the mispricing is often expressed in terms of volatility instead of price value

2. **A trader may want to construct a position that reflects a particular view of market conditions**. e.g., long gamma but not not expose to directional risk.

3. **Spreading strategies help to control risk**.

> spreading maintains profit potential but reduces short-term risk


## Chapter 13: Risk Considerations

In option trading, assuming the theoretical price is correct, the immediate reward of a trade is the captured edge. Because there is no guarantee that our theoretical price will be right, the risk associated with the trade is also introduced. The risk is from multiple dimension:

- Delta risk (usually hedged)
- Gamma risk
- Theta risk (opposite side of Gamma risk)
- Vega risk
- Rho risk (usually ignored)

So basically, in option trading, we only need to worry about two things:
- **realized volatility (underlying price) change: gamma risk**
- **implied volatility change: vega risk**

In fact, they are both **volatility risk**.

### 13.1 Volatility risk

> For an option trader, volatility risk comes in two forms—the risk that he has incorrectly estimated the **realized volatility** of the underlying contract over the life of a strategy and the risk that **implied volatility** in the option market will change.

As we mentioned before, Gamma is related to realized volatility, and Vega is related to implied volatility.

> Any spread that has a nonzero gamma or vega has volatility risk.

Steps to analysis risk for multiple kinds of spreads:

1. calculate theoretical edge
2. align theoretical edge (size up for spread with smaller edge)
3. compare risks when underlying price move

#### 13.1.1 Example

Suppose that we find the implied volatility is higher than our theoretical value (18%). If we believe in our model, we conclude that they options are **overpriced**. To make a profit, trade chooses to **short vega**. There are multiple spreads to consider (excluding calendar spreads):

- Short straddles and strangles (`\/` or `\_/`)
- Call or put ratio spreads, sell more than buy (`__/--` or `--\__`)
- Long butterflies (`--\/--`)

Below is the greeks for each spread:

![Greeks](https://upload-images.jianshu.io/upload_images/4482847-063403a282b29984.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Theoretical PnL of each spread when underlying price moves:

![Theoretical PnL v.s. Underlying Price](https://upload-images.jianshu.io/upload_images/4482847-7252dccd840da9e5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Because **we are shorting Gamma, large underlying price move will hurt our position**. But different spreads have different exposure to the risk.

Theoretical PnL of each spread when implied volatility moves:

![Theoretical PnL v.s. Volatility](https://upload-images.jianshu.io/upload_images/4482847-4eac7e37ce9e1485.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Because **we are shorting Vega, increasing of volatility will also hurt the position**.

When deciding which spread to trade, we can analysis from 2 aspects.

- **Risk**. Considering only the gamma and vega risk, Spread 3 probably has the best risk characteristics. It has **limited risk** if there is a **large move** in either direction and performs better than either Spread 1 or Spread 2 if there is a dramatic change in volatility. But if there's only **small move**, Spread 2 outperforms Spread 1 and 3.

- **Liquidity**. Spread 3 requires a lot of quantity on the 3 options. It might be not possible to execute. If we can only choose from Spread 1 and 2, Spread 2 is the clear winner.

> straddles and strangles are the riskiest of all spreads

This can be easily seen from their payoff graphs.

### 13.2 Dividends and Interest Risk

> For stock options, there are two additional risks: the risk of changing interest rates and the risk of changing dividends


The interest-rate and dividend risk associated with volatility spreads is usually small compared with the volatility (gamma and vega) risk. Nonetheless, a trader ought to be aware of these risks, especially when a position is large and there is significant risk of a change in either interest rates or dividends.


> A good spread is not necessarily the one that shows the greatest profit when things go well; it may be the one that shows the least loss when things go badly.

#### 13.3 Efficiency

> One method that traders sometimes use to compare the relative riskiness of potential strategies focuses on the risk-reward ratio, or **efficiency**, of the strategies

For example, a trader is considering two possible spreads, both with a positive gamma and negative theta. The reward coms from the Gamma and the risk is mainly from Theta. Of course the trader want the reward (Gamma) to be as large as possible compared to the risk (Theta).

The efficiency is a ratio of the absolute value:

$$\text{efficiency} = |\frac{\text{Gamma}}{\text{Theta}}|$$


We can calculate the efficiency for above 3 spreads:

| |Gamma | Theta | Efficiency |
| :-: | :-: | :-: | :-: |
| Spread 1| -406.0 | +0.4235 | 958.68 |
| Spread 2| -165.5 | +0.1365 | 1208.79 |
| Spread 3| -370.0 | +0.4000 | 925.00 |

Because we are shorting Gamma, the reward is from Theta. So the smaller efficient is, the better. As a result, from a fast comparison, we find Spread3 is the best in risk term, which is consistent with our previous analysis of each spread. (Why the spread 2 is the worst? The book does not explain...)

In cases that the **Gamma and Theta are the primary risks to the position**, Assuming that all strategies have approximately the **same theoretical edge**, the efficiency can be a reasonable method of quickly comparing strategies where all options **expire at the same time**.

#### 13.4 Delta Adjustments

Trader needs to adjust the position to remain delta-neutral.

> An adjustment to a trader’s delta position may reduce his directional risk, but if he simultaneously increases his gamma, theta, or vega risk, he may inadvertently be exchanging one type of risk for another.

Again, use above case as example.
- underlying price goes down
- since we are shorting Gamma, we will get **positive delta**
- to hedge the delta position, we can:
  - sell underlying contracts. It does not affect other Greek Risks.
  - sell calls. This will help capturing **more theoretical edge**, but also accumulate **more Greek Risks**. And the trader will have to adjust position more frequently. Besides, If the market now makes a violent move in either direction, the adverse consequences will be greatly magnified.
  - buy puts. Trader can also **reduce the Greek Risks**. But, because we think the implied volatility is overpriced, when buying puts, the trader is actually doing trade with **negative theoretical edge**.
