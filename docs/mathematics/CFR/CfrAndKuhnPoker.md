# Counterfactual Regret Minimization and Kuhn Poker

在上一篇文章 [Regret Matching and Blotto Game](https://www.jianshu.com/p/8995c4c2ac16) 中，我们用 regret matching 方法来找博弈的纳什均衡点。 regret matching 方法的局限性在于，它只适用于能用矩阵表示的博弈，即每个玩家同时采取行动的博弈，例如剪刀石头布、blotto game。

## 1 Kuhn Poker
还有一种博弈叫做”连续博弈“(sequential game)，即玩家的动作并不同时发生，而是依次发生的，例如德州扑克。它有个简化版本 Kuhn Poker，我们将用它作为例子实现 Counterfactual Regret Minimization。

> Kuhn Poker is a simple 3-card poker game by Harold E. Kuhn [8]. Two players each ante 1 chip, i.e. bet 1 chip blind into the pot before the deal. Three cards, marked with numbers 1, 2, and 3, are shuffled, and one card is dealt to each player and held as private information. Play alternates starting with player 1. On a turn, a player may either pass or bet. A player that bets places an additional chip into the pot. When a player passes after a bet, the opponent takes all chips in the pot. When there are two successive passes or two successive bets, both players reveal their cards, and the player with the higher card takes all chips in the pot.

Kuhn Poker 可以用树来表示。一共有三类节点：

1. chance node，如图中的根结点表示的发牌节点

2. decision node，玩家做决策的节点

3. terminal node，游戏结束，结算 payoff 的节点，即树中的叶子节点。

![Game tree of Kuhn Poker](https://upload-images.jianshu.io/upload_images/4482847-ac7cfad1f3fbe18c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

| Player 1 | Player 2 | Player 1 | Payoff |
| :-: | :-: | :-: | :-: |
| pass | pass |  | +1 to player with higher card |
| pass | bet | pass  | +1 to player 2 |
| pass | bet | bet  | +2 to player with higher card |
| bet | pass |  | +1 to player 1 |
| bet | bet |  | +2 to player with higher card |

图中的要点：

- 每个节点代表一个状态，节点与节点的边代表 action。状态就是 information set，即__做决策时__所有可以获取的信息（包括历史 game 信息）
- 每个节点有两个分支，分别代表跟牌 (pass) 和加注 (fold)。
- 如果在对方 bet 后选择 pass 则认定为负
- 如果都 bet 或者都 pass，牌大的胜

Kuhn 共有 12 种 information set，对手牌 1，2，3 各有 4 种：

1. 本方先手，决定 pass 还是 bet
2. 本方先手 pass，对手 bet，再次决定 pass 还是 bet
3. 本方后手，对方 pass，决定 pass 还是 bet
4. 本方后手，对方 bet，决定 pass 还是 bet

由于我们无法得知对方手牌，每种 information set 包括了 2 个 game state。因为总共有 3 张牌，对手手牌可能是除去我们手牌外的 2 张任意一张。


这也说明了 information set 与 game state 的区别。game state 是客观存在的状态，而 information set 是主观观察到的信息，它不一定是完整的，因此与 game state 可能是一对多的关系。


## 2 Counterfactual Regret Minimization

> “Counterfactual” means contrary to fact. Intuitively, the counterfactual regret closely measures the **potential gain** if we do something contrary to the fact, such as deliberately making action $a$ at information set $I$ instead of following strategy $\sigma^t$.

### 2.1 Notation

| symbol | term | comment |
| :-: | :-: | :-: |
| $h$ | action history | from root of the game |
| $\sigma_i$ | player $i$'s strategy | probability of choosing action $a$ in information set $I$ |
| $\sigma$ | strategy profile | all player strategies together |
| $\pi ^ {\sigma} (h)$| reach probability | reach probability of game history $h$ with strategy profile $\sigma$ |
| $u$ | utility | payoff |
| $t$ | time step | every information set has an independent $t$, it is incremented with each visit to the information set |

### 2.2 Formula

Let $Z$ denote the set of all terminal game histories (sequence from root to leaf). Then proper prefix $h \sqsubset z$ for $z \in Z$ is a nonterminal game history. **Z are the all possible endings from h**.

Define the **counterfactual value** at non-terminal history $h$ for player $i$ as :

$$ v_i(\sigma, h) = \sum_{z \in Z, h \sqsubset z } \pi_{-i}^{\sigma} (h) \pi ^{\sigma} (h, z) u_i (z) $$

$\pi_{-i}^{\sigma} (h)$ is the probability of reaching $h$ with strategy profile $\sigma$ excluding the randomness of player $i$'s actions (player $i$ has probability 1.0 to take current actions).

$\pi^{\sigma}(h, z)$ is the probability of reaching $z$ from $h$ with strategy profile $\sigma$.

![understanding counterfactual value](https://upload-images.jianshu.io/upload_images/4482847-19c86b8240f79a69.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

In the graph above, because $i = 0$, the p0 player has 100% probability to choose action $a_2$ and $a_4$. That's why the probability of reaching $h$ is $P(a_1)P(a_3|a_2)$.

> We treat the computation as if player $i$'s strategy was modified to have intentionally played to information set $I$. Put another way, we exclude the probabilities that factually came into player $i$’s play from the computation.

The **counterfactual regret** of not taking action $a$ at **history** $h$ is defined as:

$$r(h, a) = v_i(\sigma_{I \rightarrow a}, h) - v_i(\sigma, h)$$

$\sigma_{I \rightarrow a}$ denotes a profile equivalent to $\sigma$, except that action $a$ is always chosen at information set $I$.

Since **an information set may be reached through multiple game histories**, the counterfactual regret of not taking action $a$ at **information set** $I$ is:

$$ r(I, a) = \sum_{h \in I} r(h, a) $$

Let $r_i^t(I, a)$ refer to the regret in time $t$ belonging to player $i$, the **cumulative** counterfactual regret is defined as:

$$ R_i^T(I, a) = \sum_{t=1}^T r_i^t(I, a) $$

For each information set $I$, the probability of choosing action $a$ is calculated by:

$$
\sigma^{T+1}_i (I, a) = \begin{cases}
\frac{R_i^{T, +}(I, a)}{ \sum_{a \in A(I)} R_i^{T, +}(I, a) } & \text{if} \sum_{a \in A(I)} R_i^{T, +}(I, a) > 0\\
\frac{1}{|A(I)|} & \text{otherwise.}
\end{cases}
$$

$+$ means positive (>0). It selects actions in proportion to positive regrets. This is the same as the **regret matching** algorithm in [Regret Matching and Blotto Game](https://www.jianshu.com/p/8995c4c2ac16).


### 2.3 Algorithm

CFR 参数：

1. action history: $h$
2. learning player id: $i$
3. time step: $t$
4. reach probability of action history $h$ for player 1: $\pi_1$
5. reach probability of action history $h$ for player 2: $\pi_2$

![Counterfactual Regret Minimization Algorithm](https://upload-images.jianshu.io/upload_images/4482847-f2a29b5c86d0da9a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这个算法其实写的比较晦涩，参数设计也不是特别合理。实际上，我们只需要三个参数：

- 游戏历史 `history`
- 玩家的手牌 `cards`
- 每个玩家到达这个历史节点的概率 `reach_probs`

当前玩家 id 可以通过游戏历史来推断。

我自己的实现如下：

```rust
    fn cfr(
        &mut self,
        history: kuhn::ActionHistory,
        cards: &Vec<i32>,
        reach_probs: HashMap<i32, f64>,
    ) -> f64 {
        // current active player
        let player_id = (history.0.len() % 2) as i32;

        let maybe_payoff = kuhn::get_payoff(&history, cards);
        if maybe_payoff.is_some() {
            let payoff = match player_id {
                0 => maybe_payoff.unwrap(),
                1 => -maybe_payoff.unwrap(),
                _ => panic!("unexpected player id {}", player_id),
            };
            return payoff as f64;
        }

        // not the terminal node
        let info_set = InformationSet {
            action_history: history.clone(),
            hand_card: cards[player_id as usize],
        };

        if self.cfr_info.contains_key(&info_set) == false {
            self.cfr_info.insert(info_set.clone(), CfrNode::new());
        }

        let action_probs = self
            .cfr_info
            .get(&info_set)
            .unwrap()
            .get_action_probability();

        let mut action_payoffs = Vec::new();
        let mut node_value = 0.0;
        for (action_id, action_prob) in action_probs.iter().enumerate() {
            // next history, appending the new action to it
            let mut next_history = history.clone();
            next_history
                .0
                .push(kuhn::Action::from_int(action_id as u32));
            // update history probability
            let mut next_reach_probs = reach_probs.clone();
            *next_reach_probs.get_mut(&player_id).unwrap() *= action_prob;

            // recursive call, "-" here because the return value is the opponent's payoff
            action_payoffs.push(-self.cfr(next_history, cards, next_reach_probs));
            node_value += action_prob * action_payoffs[action_id];
        }

        assert_eq!(action_payoffs.len(), 2);

        // update regret
        let node = self.cfr_info.get_mut(&info_set).unwrap();
        for (action_id, payoff) in action_payoffs.iter().enumerate() {
            let regret = payoff - node_value;
            let opponent = 1 - player_id;
            node.cum_regrets[action_id] += reach_probs[&opponent] * regret;
        }

        for (action_id, action_prob) in action_probs.iter().enumerate() {
            node.cum_strategy[action_id] += reach_probs[&player_id] * action_prob;
        }

        return node_value;
    }
```

算法中的 chance node 其实就是类似于发牌，掷骰子这类随机节点。在 kuhn 中仅有一个随机节点，就是开局的发牌环节。因此我们可以提前随机发好牌，不用在 cfr 算法内处理。

这是一种多叉树的深度优先遍历。每个 information set 就是一个节点。每个节点采取的不同 action 会通向下一个子节点，直到叶子节点（即游戏结束，有具体 payoff 值的节点）。

核心思路：

1. 每个节点采取的 action 可以通向下一个节点。
2. 对于每个节点，节点价值的计算是所有通过 action 可以到达的子节点价值的加权平均。权重为 action 的选取概率。
3. action的选取概率根据 regret matching 算法确定，若 regret 尚未初始化就随机选择 action
4. action 的 regret 可以由选取该 action 到达的子节点价值减去当前节点价值得到。
5. 每次迭代，将 action 的 regret 乘上到达该节点概率 (`reach_probs[&opponent]`)，累计到这个节点这个 action 的 `cum_regret` 中。这里采用 __对手方__ 到达概率的原因是己方 action 我们都是以 1.0 的概率选择的。

“反事实”体现在每个节点，虽然事实上我们只能选择一个行动，但是我们可以虚拟地尝试所有的行动。

此外，最后趋近于 Nash Equilibrium 的不是我们的 regret matching 策略，而是累计策略（`cum_strategy`），计算方式见代码末尾。

### 2.4 结论

有一些简单的验证方法：

1. Kuhn Poker 的第一位玩家每局收益期望是 -1/18，因为第二位玩家有信息上的优势（知道第一位玩家的行动）。
2. 玩家 1 拿到手牌 1 选择 Bet 的概率应该在 (0.0, 1/3) 之间（原因见 Q & A）
3. 玩家 1 拿到手牌 2 应该永远选择 Check，因为如果选择 bet，对方手牌是 3，必定 bet，输 \$ 2, 如果对方手牌是 1，必定也 check，赢 \$1，与 check 赢的钱一致，但是额外承受风险。
4. 玩家 1 拿到手牌 3 选择 Bet 的概率应该是手牌 1 选择 bet 概率的 3 倍（原因见 Q & A）

各 information set 选取 check 或者 bet 的概率（information set表示为 手牌 + 当前 action history）：

| History | Check Probability | Bet Probability |
| :-: |  :-: |  :-: |
| 1 | 0.77 | 0.23 |
| 1C | 0.67 | 0.33 |
| 1B | 1.00 | 0.00 |
| 1CB | 1.00 | 0.00 |
| 2 | 1.00 | 0.00 |
| 2C | 1.00 | 0.00 |
| 2B | 0.66 | 0.34 |
| 2CB | 0.42 | 0.58 |
| 3 | 0.30 | 0.70 |
| 3C | 0.00 | 1.00 |
| 3B | 0.00 | 1.00 |
| 3CB | 0.00 | 1.00 |

### 2.5 一些难点

本节主要以 Q & A 的形式阐明一下普遍会遇到的困惑。

#### 2.5.1 为什么需要考虑 reach probability

考虑 reach probability 才能算出动作的期望收益，从而计算最佳应对（best response）。

用一个例子说明。假设现在我们手牌是 2，对方 Bet，我们无法知道对方手牌是 1 还是 3。但是我们知道：

- 如果对方手牌是1，我们跟 Bet，payoff = 2，如果选择 Check，payoff = -1
- 如果对方手牌是3，我们跟 Bet，payoff = -2，如果选择 Check，payoff = -1

假设对手以手牌 1 Bet 的概率（即 history= “1B” 的 reach probability）为 $a$，以手牌 3 Bet 的概率（即 history = “3B” 的 reach probability）是 $b$。我们可以得到 Check 的期望收益：

$$ \text{EV}_{Check} =  - a - b $$

同理，Bet 的期望收益：

$$ \text{EV}_{Bet} =  2a - 2b $$

我们肯定倾向于选择期望收益大的行动。即，若 $b > 3a$，则应该选择 Check，反之则选择 Bet。在 Nash Equilibrium 情况下，这两个动作的期望收益应该相等，即 $b = 3a$，此时我们无法exploit 我们的对手，当然，对手也无法 exploit 我们。这就是之前验证结果时的 “3 倍” 的原因。


#### 2.5.2  为什么需要维护一个 strategy sum

在每次迭代中，cumulative regret 的值不是很稳定，一些重要的 action 可能恰好在 0.0 附近摇摆，如果选用它来求最终的策略，可能导致有的动作无法被选择。

使用 strategy sum 就不存在这个问题，它永远是正值，并且数学上收敛到 Nash Equilibrium。


## 2.6 CFR 算法的不足

- 它需要遍历整个游戏树

- 它需要知道对手的策略，这在实际情况下很难满足

针对这两个缺点，Marc Lanctot 等人提出了 Monte-Carlo CFR。我们将在下一篇文章中介绍。

## Reference

1. [Building a Poker AI](https://ai.plainenglish.io/building-a-poker-ai-part-6-beating-kuhn-poker-with-cfr-using-python-1b4172a6ab2d)

2. [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://papers.nips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf)
