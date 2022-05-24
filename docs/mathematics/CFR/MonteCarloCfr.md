# Monte Carlo CFR

在上一篇文章中，我们讲了经典的 CFR 算法的原理及实现。但是，它也有两个致命的缺陷，导致无法应用到实际的复杂博弈中：

- 它需要遍历整个游戏树

- 它需要知道对手的策略，这在实际情况下很难满足

本文将介绍基于 Monte Carlo 的改进算法。全部代码可以参考我的 github 项目：[cfr-kuhn](https://github.com/double-free/CfrKuhnPoker).

## Definition

核心思想是，在避免遍历整个游戏树的前提下，每个 information set 的每个 action 的 counterfactual regret 期望值保持不变。

令 $\mathcal{Q}  =  \{Q_1, Q_2, ..., Q_r \}$ 是所有 terminal node $Z$ 的子集，其中的每一个元素 $Q_j$ 我们称为 __block__，每个 block 可能包含多个 terminal node 。令 $q_j > 0$ 是选择 $Q_j$ 的概率，满足 $\sum_{j=1}^r q_j = 1$。每次迭代，我们都会从这些 blocks 中采样一个并且只考虑在该 block 中的 terminal node 。

回忆 CFR 经典版的 counterfactual value 计算公式：

$$ v_i(\sigma, I) = \sum_{z \in Z_I } \pi_{-i}^{\sigma} (z[I]) \pi ^{\sigma} (z[I], z) u_i (z) $$

在 MCCFR 中，对于 block $Q_j$ 的 sampled counterfactual value 可以表示为：

$$v_i(\sigma, I | j) = \sum_{z \in Q_j \cap Z_I} \frac{1}{q(z)} \pi_{-i}^\sigma (z[I]) \pi^\sigma (z[I], z) u_i(z)$$

其中：

- $z$ 是 terminal node
- $z_I$ 表示 information set $I$ 可以到达的 $z$ 的全集
- $z[I]$ 表示某个可以到达 $z$ 的 information set
- $i$ 表示玩家 id
- $j$ 表示 block id
- $q(z)$ 表示 $z$ 所在 block 被选中的概率之和（一个 terminal node 可以被分到多个 block 中）


我们可以证明 counterfactual value 和 sampled counterfactual value 期望是相等的。

这就是 MCCFR 的基本思想了。事实上，CFR 可以看作 MCCFR 在 $\mathcal{Q} = \{Z\}$ 且 $q_1 = 1.0$ 的特例。

此外，我们可以__将每一个 block 选为 chance node 的一个分支__。以 Kuhn Poker 为例，我们对于手牌 1，2，3 可以建立 3 个 block： $\{ Q_1, Q_2, Q_3 \}$，分别包含手牌为 1，2，3 的所有 terminal nodes。从而，每个 block 的概率也自然为 $\frac{1}{3}, \frac{1}{3}, \frac{1}{3}$。这就是 chance-sampled CFR。


### Outcome-Sampling MCCFR

在 outcome-sampling MCCFR 中，我们__把每个 terminal node 作为一个 block__。每次迭代，我们抽取一个 terminal node 并更新它的前缀 information set。一个 terminal node 出现概率越高，我们采样概率也越高。因此我们必须得到每个 terminal node j 出现的概率作为 $q_j$。如何得到呢？

在 CFR 中，我们在计算 regret 的时候，会计算到达每个 history 的概率。对于 terminal node j，这个概率其实就是采样概率 $q_j$。

CFR 算法包含了两个方向的遍历：

- 前向遍历。用于计算每个玩家从原点到达当前 history 的概率 $\pi_i^{\sigma}(h)$
- 后向遍历。用于计算每个玩家从当前 history 进行到 terminal node 的概率 $\pi_i^{\sigma}(h,z)$，在此过程中，还会计算 sampled counterfactual regrets。

后向遍历中计算 sampled counterfactual regrets 会分为两种情况。

其一是选择了能通向当前 terminal node $z$ 的 action。即，在information set $I$ 采取了行动 $a$，此时计算方式为：

$$ \begin{align}
r(I, a) &= v_i(\sigma_{I \rightarrow a}, I) - v_i(\sigma, I) \\
          &= \frac{\pi_{-i}^\sigma (z[I]) \pi^\sigma (z[I]a, z) u_i(z)}{q(z)} - \frac{\pi_{-i}^\sigma (z[I]) \pi^\sigma (z[I], z) u_i(z)}{q(z)} \\
          &= \frac{\pi_{-i}^\sigma (z[I]) \pi^\sigma (z[I] a, z) u_i(z)}{q(z)} ( 1 - \sigma(a|z[I]))
\end{align} $$

上式比较难以理解的是：

$$\pi^\sigma (z[I] a, z) \cdot \sigma(a|z[I]) = \pi^\sigma (z[I], z) $$

结合定义，$\pi^\sigma (z[I] a, z)$ 是从 $I + a$ 到达 $z$ 的概率，相比 $\pi^\sigma (z[I], z)$ 多前进了一步（选择行动 $a$），在 $I$ 选择 $a$ 的概率是 $\sigma(a|z[I])$，因此，如果从 $I$ 到 $z$ 的概率为 $p$，由于 $I + a$ 将选择  $a$ 的概率由原本的 $\sigma(a|z[I])$ 变成了 1.0，它到 $z$ 的概率就变成了 $p / \sigma(a|z[I])$。

另一种情况是选择了其他行动，此时 $I + a$ 不是 $z$ 的前缀。此时的 sampled counterfactual value 是：

$$ \begin{align}
r(I, a) &= 0 - v_i(\sigma, I) \\
          &= - \frac{\pi_{-i}^\sigma (z[I]) \pi^\sigma (z[I] a, z) u_i(z)}{q(z)} \sigma(a|z[I])
\end{align} $$

选取 0 是因为该动作的后悔值更新不归 $z$ 管。必然有别的 terminal node 的前缀是 $I + a$，当采样到这些 terminal node 的时候自然会更新。


我们令：

$$ w_I =  \frac{\pi_{-i}^\sigma (z[I]) \pi^\sigma (z[I] a, z) u_i(z)}{q(z)} $$

由于在 outcome sampling 中，每个 terminal node $z$ 都对应一个 block $Q$，因此 $q(z)$ 就是选中 $Q$ 的概率。理论上，为了保证采样的真实性，这个概率应该与所有玩家遵循策略 $\sigma$ 进行游戏到达 $z$ 的概率相同，即

$$q(z) = \pi^\sigma(z)$$

根据定义，$\pi_{-i}^\sigma (z)$ 是对手到达 $z$ 的概率，在计算时，假设玩家 $i$ 以 1.0 的概率选择动作，而 $\pi_i^\sigma (z)$ 相反。两者的乘积就是从局外旁观者看来到达 $z$ 的概率。因此有：

$$ \begin{align}
q(z) &= \pi^\sigma(z) \\
 &= \pi^\sigma(z[I]) \cdot \pi^\sigma(z[I], z) \\
 &= \pi_i^\sigma (z[I]) \cdot \pi_{-i}^\sigma (z[I]) \cdot \pi^\sigma(z[I], z)
\end{align} $$

带入得到：

$$\begin{align}
w_I &= \frac{\pi_{-i}^\sigma (z[I]) \pi^\sigma (z[I] a, z) u_i(z)}{q(z)} \\
       &= \frac{\pi^\sigma (z[I] a, z) u_i(z)}{ \pi_i^\sigma (z[I]) \cdot \pi^\sigma(z[I], z)} \\
       &= \frac{\pi_i^\sigma(z[I]a, z) u_i(z)}{\pi_i^\sigma(z)} \\
       &=  \frac{u_i(z)}{ \pi_i^\sigma (z[I]) \cdot \sigma(a|z[I])}
\end{align}$$

注意在这个表达式中，我们不再需要对手的信息 （只有带下标 $i$ 的部分了）。我们在化简中刻意消去了对手 $-i$ 相关的信息。

每次迭代累计后悔值会增加：

$$ r(I,a) = \begin{cases}
w_I \cdot (1 - \sigma(a|z[I]) & \text{if} ~ z(z[I]a) \sqsubset z \\
-w_I \cdot \sigma(a|z[I]) & \text{otherwise}
\end{cases}$$

这样计算可以保证累计后悔值的期望是与传统 cfr 相同的。

每次迭代，每个动作的累计 strategy 增加：

$$s(I, a) = \pi_i^\sigma(z[I]) \cdot \sigma(a|z[I]) $$

> 注：论文中还增加了 $t - c_I$ 权重，即本次 iteration 和上次访问 $I$ 的 iteration 之差，但是我加上权重后无法得到正确结果，尚需进一步研究。

### 实现细节

相比 cfr，mccfr 的不同主要有：
1. 无需对手（-i）的信息
2. 无需遍历所有行动
3. 计算后悔值的方法不同，需要增加一个基于采样概率的权重。
4. 需要引入 epsilon 保证探索所有 action

```rs
    fn mccfr(
        &mut self,
        history: kuhn::ActionHistory,
        cards: &Vec<i32>,
        reach_probs: HashMap<i32, f64>,
    ) -> f64 {
        // current active player
        let player_id = (history.0.len() % 2) as i32;
        let opponent_id = 1 - player_id;

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
            self.cfr_info.insert(info_set.clone(), CfrNode::new(0.06));
        }

        let action_probs = self
            .cfr_info
            .get(&info_set)
            .unwrap()
            .get_action_probability();

        let chosen_action_id = sample(&action_probs);
        let chosen_action = kuhn::Action::from_int(chosen_action_id);
        let chosen_action_prob = action_probs[chosen_action_id as usize];
        let mut next_history = history.clone();
        next_history.0.push(chosen_action);
        // modify reach prob for SELF (not opponent)
        // update history probability
        let mut next_reach_probs = reach_probs.clone();
        *next_reach_probs.get_mut(&player_id).unwrap() *= chosen_action_prob;
        // recursive call
        // final payoff of the terminal node
        let final_payoff = -self.mccfr(next_history, cards, next_reach_probs);

        // update regret value
        let node = self.cfr_info.get_mut(&info_set).unwrap();
        for (action_id, action_prob) in action_probs.iter().enumerate() {
            let action = kuhn::Action::from_int(action_id);
            // reach probability of SELF (not opponent)
            let weight = final_payoff / reach_probs[&player_id] / action_prob;
            if action == chosen_action {
                node.cum_regrets[action_id] += weight * (1.0 - action_prob);
            } else {
                node.cum_regrets[action_id] += -weight * action_prob;
            }
        }

        // update strategy
        for (action_id, action_prob) in action_probs.iter().enumerate() {
            node.cum_strategy[action_id] += action_prob * reach_probs[&player_id];
        }

        return final_payoff;
    }
```

### 结果

Explore with `epsilon=0.06`. Average payoff = -0.05138

| History | Check Probability | Bet Probability |
| :-: |  :-: |  :-: |
| 1 | 0.805 | 0.195 |
| 1C | 0.68 | 0.32 |
| 1B | 0.97 | 0.03 |
| 1CB | 0.97 | 0.03 |
| 2 | 0.97 | 0.03 |
| 2C | 0.94 | 0.05 |
| 2B | 0.56 | 0.44 |
| 2CB | 0.47 | 0.53 |
| 3 | 0.422 | 0.578 |
| 3C | 0.03 | 0.97 |
| 3B | 0.03 | 0.97 |
| 3CB | 0.03 | 0.97 |

We get 0.03 here because we choose `epsilon = 0.06` and we have 2 actions to choose from.

## Reference

1. [Monte Carlo Counterfactual Regret Minimization](https://blog.csdn.net/qq_36691985/article/details/116793223)
2. [MCCFR technical report](https://era.library.ualberta.ca/items/944add23-5c61-4b7a-afa5-8059479a0443/view/b8226187-17cd-42bb-a299-09cae015d415/TR09-15.pdf)
