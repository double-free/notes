# Martingale


## Definition
Process X is a martingale if for all n:

$$ E[X_{n+1}|F_n] = X_n $$

Where

- Fn is the history of Xn (called filtration)

Which means, on the n+1 step, the expectation of X shall be the same as any step before.

 A martingale may be thought of as a “fair game”, because given the information in current and previous plays (Fn), you don't expect to change your total winning (X).

## Example

Here are two practical example to help you understand it.

### Partial sum process

A simple coin game, in $i$ turn we bet $X_i$, and $S_n$ is our winning money after n turn.

Let Xi be independent, define the partial sum process:

$$\begin{aligned}
S_0 &= 0, \\
S_n &= \sum_{ i=1 }^n X_i, n=1, 2,... \end{aligned}$$

Sn is a martingale iff:

$$E(X_i) = 0$$

#### Proof

With Xi being independent, and E(Xi) = 0, we have:

$$\begin{aligned}
E[S_{n+1}|X_1,...X_n] &= E[S_n + X_{n+1} | X_1,...X_n] \\
&= S_n + E[X_{n+1} | X_1,...X_n] \\
&= S_n + E[X_{n+1}] \\
&= S_n
\end{aligned}$$

### Gambler's Ruin Problem

A classic problem on martingale.

> Consider a gambler who starts with an initial fortune of \$1 and then on each successive gamble either wins \$1 or loses \$1 independent of the past with probabilities p and q = 1−p respectively. Let Rn denote the total fortune after the n th gamble. The gambler’s objective is to reach a total fortune of \$N, without first getting ruined (running out of money). If the gambler succeeds, then the gambler is said to win the game.

Let $P_i$ denote the probability that the gambler wins when the initial money $R_0 = i$, we have:

$$P_i = pP_{i+1} + qP_{i-1}$$

This is because P_i can only lead to two states:

*   Winning \$1 with probability p to state $P_{i+1}$ 
*   Losing \$1 with probability q to state $P_{i-1}$

Subtract P_{i-1} from both sides of the equation, we get:

$$P_i = (p + q) P_i = pP_{i+1} + qP_{i-1}$$

i.e.,

$${P_{i+1} - P_i}= \frac{ q }{ p }(P_i - P_{i-1})$$

Thus

$$\begin{aligned} P_{i+1} - P_1 &= \sum_{k=1}^i (P_{k+1} - P_k) \\ &= \sum_{k=1}^i (\frac{q}{p})^k P_1 \end{aligned}$$

We have

$$P_{i+1}= \begin{cases} P_1 \frac{1-(q/p)^{i+1}}{1-(q/p)} &, p \neq q\\ P_1 (i+1) &, p=q \end{cases}$$

To solve P_1, pick i = N-1 and use the fact that P_N = 1

$$P_1= \begin{cases} \frac{1-(q/p)}{1-(q/p)^N} &, p \neq q\\ 1/N &, p=q \end{cases}$$

Substitute P_1 and we have:

$$P_i= \begin{cases}
\frac{ 1-(q/p)^i}{ 1-(q/p)^N } &, p \neq q \\
i/N &, p=q
\end{cases}$$

## Reference

1.  [martingales.dvi (rice.edu)](http://www.stat.rice.edu/~dcox/Stat581/martingales.pdf)
2.  [4700-07-Notes-GR.pdf (columbia.edu)](http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GR.pdf)
