# Regret Matching and Blotto Game


## 1 基本概念
2000 年，Hart 和 Mas-Colell 介绍了一个重要的博弈论算法 regret matching。博弈双方通过：

1. 记录后悔值
2. 根据后悔值的比例地选择下一步行动

达到纳什均衡 (Nash equilibrium)。这个能很好地解决正则形式的博弈（normal-form game），但是对扩展形式的博弈（extensive-form game）不适用。

所谓正则形式，是一种描述博弈的方式。正则形式用 n 维矩阵来描述博弈，而扩展形式使用图。正则形式只能描述玩家同时行动的博弈。

### 1.1 博弈的正则形式描述

正则形式的矩阵描述有以下几个要素：
1. 玩家数量 n 即维度
2. 维度 i 上的值是玩家 i 的行动
3. 矩阵的元素是奖励( payoff )

例如，我们可以用如下矩阵来描述剪刀石头布游戏：

|  | 剪刀 | 石头 | 布 |
| :-: | :-: | :-: | :-: |
| 剪刀 | (0, 0) | (-1, 1) | (1, -1) |
| 石头| (1, -1) | (0, 0) | (-1, 1) |
| 布 | (-1, 1) | (1, -1) | (0, 0) |

如果矩阵中所有 payoff 的值的和为 0，则称为零和博弈。

### 1.2 玩家策略

如果某个玩家以 100% 的概率采取一个行动 （例如德扑全程 all in），称为 pure strategy。如果一个玩家可能采取多种行动，就称为 mixed strategy。

我们使用 $\sigma$ 表示 mixed strategy，$\sigma_i(s)$ 表示玩家 $i$ 选择行动 $s$ 的概率，$-i$ 表示 $i$ 的对手。

我们可以通过以下方法计算玩家的期望 payoff：

$$u_i(\sigma_i, \sigma_{-i}) = \sum_{s \in S_i}\sum_{s' \in S_{-i}} \sigma_i(s) \sigma_{-i}(s')u_i(s, s')$$

其实就是加权求和。


## 2 Regret Matching and Minimization

Regret matching 算法只能用于正则形式的博弈。其基本思想为根据 payoff 对之前的行动作求反悔值。再利用累计的反悔值指导下一步行动。

以刚才的石头剪刀布游戏为例，我们出了石头，对手出了布，我们输掉了 1 块钱，payoff 是 -1。我们如果出布，是平局，payoff 是 0，如果出剪刀，就赢了，payoff 是 1。payoff 差值即反悔值分别是 (0，1，2)。将其 normalize，下一次出石头、布、剪刀的概率分别是 (0，1/3，2/3)。

假设第二次我们出了剪刀，对手出了石头。我们再次输掉了 1 块钱。这次我们对石头、布、剪刀的反悔值分别是(1，2，0)。累加到之前的 (0，1，2) 上为 (1，3，2)。下一次出石头、布、剪刀的概率为 (1/6, 3/6, 2/6)。

### Exercise: Colonel Blotto

> Colonel Blotto and his arch-enemy, Boba Fett, are at war. Each commander has S soldiers in total, and each soldier can be assigned to one of N < S battlefields. Naturally, these commanders do not communicate and hence direct their soldiers independently. Any number of soldiers can be allocated to each battlefield, including zero. A commander claims a battlefield if they send more soldiers to the battlefield than their opponent. The commander’s job is to break down his pool of soldiers into groups to which he assigned to each battlefield. The winning commander is the one who claims the most battlefields. For example, with (S,N) = (10,4) a Colonel Blotto may choose to play (2,2,2,4) while Boba Fett may choose to play (8,1,1,0). In this case, Colonel Blotto would win by claiming three of the four battlefields. The war ends in a draw if both commanders claim the same number of battlefields.

让两个使用 regret matching 的玩家进行对战，选定 S=5 和 N=3，找到 Nash Equilibrium。

要点：

1. 列出所有分配方法，作为可能的 action set，从 0 开始依次编号
2. 用一个 2 维矩阵存储 action 对 action 的胜负表，即得到 blotto 博弈的矩阵描述
3. 各个玩家根据在具有正反悔值的 action 中，根据反悔值随机选择一个 action。若不存在正反悔值的 action（例如第一轮），则随机选择初始 action。
4. 各个玩家在游戏结束时得到对手的 action，并更新自己的反悔值

代码实现见 https://github.com/double-free/cfr ，运行 10 万次结果如下（结果具有一定随机性）。

可能的分配方式：
```console
[0, 0, 5],
[0, 1, 4],
[0, 2, 3],
[0, 3, 2],
[0, 4, 1],
[0, 5, 0],
[1, 0, 4],
[1, 1, 3],
[1, 2, 2],
[1, 3, 1],
[1, 4, 0],
[2, 0, 3],
[2, 1, 2],
[2, 2, 1],
[2, 3, 0],
[3, 0, 2],
[3, 1, 1],
[3, 2, 0],
[4, 0, 1],
[4, 1, 0],
[5, 0, 0],
```

每个分配方式对应的反悔值：

```console
player 1: for opponent 0, regret sum for each action [-55002, -10612, -45, -41, -11250, -55644, -10426, 25, -11075, 127, -11068, 109, -11107, -11009, 207, -74, -92, 20, -11636, -11640, -56029]
```

最终还具有正反悔值的 action：
```console
player 1: for opponent 0, candidate strategy Allocation { soldiers: [1, 1, 3] } has positive regret 25
player 1: for opponent 0, candidate strategy Allocation { soldiers: [1, 3, 1] } has positive regret 127
player 1: for opponent 0, candidate strategy Allocation { soldiers: [2, 0, 3] } has positive regret 109
player 1: for opponent 0, candidate strategy Allocation { soldiers: [2, 3, 0] } has positive regret 207
player 1: for opponent 0, candidate strategy Allocation { soldiers: [3, 2, 0] } has positive regret 20
```

结果还是相对符合直觉的。

## Reference

1. [反事实后悔最小化](https://zhuanlan.zhihu.com/p/339612936)
2. [An Introduction to Counterfactual Regret Minimization](https://www.ma.imperial.ac.uk/~dturaev/neller-lanctot.pdf)
3. [Game Basics](https://xyzml.medium.com/learn-ai-game-playing-algorithm-part-i-game-basics-46b522cda88b)
4. [Monte Carlo Tree Search](https://xyzml.medium.com/learn-ai-game-playing-algorithm-part-ii-monte-carlo-tree-search-2113896d6072)
5. [Counterfactual Regret Minimization](https://xyzml.medium.com/learn-ai-game-playing-algorithm-part-iii-counterfactual-regret-minimization-b182a7ec85fb)
