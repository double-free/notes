# ESL 10.9: Boosting Trees

“off-the-shelf“ 的方法是可以无需经过复杂的预处理（如归一化），也无需仔细调参即可用于数据训练和预测的方法。

实际问题中的数据往往不像例子中那么简单。通常，输入可能是不同类型的组合（数字、类别、bool 等），还存在丢失部分数据等问题。而决策树天生就具备处理这些输入的能力。因此，在所有已知的学习方法中，决策树是最接近 ”off-the-shelf“ 的方法。

决策树的唯一问题就是不够精确，但是如果我们将 Boosting 方法与决策树结合，就可以大大提高它的精确度。

## Regression Tree

我们在 [9.2 Tree-Based Methods](https://www.jianshu.com/p/28e861bab1e1) 中已经介绍了回归树。回归树可以表示为：

$$ T(x; \Theta) = \sum _ {j=1} ^ J \gamma_j I( x \in R_j) $$

其中：

- $x$：特征值
- $J$：总区域（叶子节点）数
- $R_j$：第 j 个区域
- $\gamma_j$：第 j 个区域的预测值
- $I$：符合条件则为 1， 否则为 0
- $\Theta$：所有参数的组合，即 J 组 $\{ R_j, \gamma_j \}$ pair，代表了最终训练出来的模型

如 9.2 中所述，对于给定训练数据，最优的树是可以确定的。因此 $J$ 是可以通过训练数据得到的。在 J 已知情况下，我们需要求解优化问题来得到 $\Theta$：

$$ \hat{\Theta} = \arg \min _{\Theta} \sum_{j=1}^J \sum_{x_i \in R_j} L(y_i, \gamma_j) $$

即，找到 $\Theta$ 使各区域损失函数 $L$ （例如 MSE）之和最小。

这是一个难以求解的组合优化问题，我们把这个求解过程分为两个部分，并寻求一个较优解而非最优解：

- （容易）给定 $R_j$ 求 $\gamma_j$：一般来讲直接使用落在$R_j$ 内的所有样本的均值 $\overline y_j$。
- （困难）确定 $R_j$：通常我们采用自顶向下的贪婪算法来递归地分区 。我们在 [9.2 Tree-Based Methods](https://double-free.github.io/notes/mathematics/ESL/ESL9_TreeBasedMethods/) 有详细的介绍。


## Boosting Trees

现在我们已经可以知道单个树的求解了。Boosting Trees 本质上是多个树相加得到：

$$ f_M(x) = \sum_{m=1}^M T(x;\Theta_m) $$

其中 M 是树的总数。

树的求和可能比较难直观理解。实质上非常简单，就是针对 __每个样本__，对其在每个树的值求和。

![树的求和](images/10/add_trees.png)

优化的目标就是找到一组树，使得所有样本的预测误差最小。其中预测误差用损失函数 $L$ 计算：

$$ L(f) = \sum_{i=1}^N L(y_i, f(x_i)) $$

我们采用 additive 的方式来求解这组树，每一步只求解一个树。基于前 k-1 个树的模型 $f_{k-1}(x)$，第 k 个树选择一个能最小化全局损失函数的树：

$$ \hat{\Theta}_k = \arg \min _{\Theta_k} \sum_{i=1}^N L(y_i, f_{k-1}(x_i) + T(x_i; \Theta_k)) $$

我们可以使用  __最速下降法__ 实现。

对于第 k 步的下降方向，我们可以选择第 k-1 步的负梯度方向。训练样本 i 的梯度方向为：

$$g_{ik} = \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} | _{f= f_{k-1}}$$

下降步长是以下优化问题的解：

$$ \rho_k = \arg \min_{\rho} (f_{k-1} - \rho g_k)$$

于是可以得到：

$$ f_k = f_{k-1} - \rho_kg_k$$

这就是 Gradient Boosting 了。

### Gradient Boosting Trees

> Gradient Boosting 的基本思想是：串行地生成多个弱学习器，每个弱学习器的目标是拟合先前累加模型的损失函数的负梯度， 使加上该弱学习器后的累积模型损失往负梯度的方向减少。

回想上面树的加法的图，每一颗树其实就是一个简单的模型（弱学习器），假设某个样本的真实值为 10，第一个模型拟合结果是 7，则误差为 7-10 = -3，第二个模型则以 3 为拟合目标，以此类推。

我们一般使用梯度下降是用来调整“参数误差”。对于 Boosting Trees，如果我们不把树看成参数而是看作模型，也可以认为实际上我们在调整“模型误差”，即通过添加新的树来修正模型。

#### 参数

$M$ - 总共的树的数量

$D$ - 每个树的最大深度

$\rho$ - 步长，即学习率

> 为什么需要设置树的最大深度？

的确，在上面 Regression tree 的生成中，我们是直接寻找“最优树”。并不限制节点和深度，而是先构造一个足够大的树，通过“剪枝”操作来得到合适的节点数。

由于 boosting trees 包含多个树，我们对每个树并不要求“最优”，设置一个最大深度可以极大地简化计算。

#### 算法

1. 初始化一个仅有一个节点的树，包含所有样本，节点的值$\gamma$为以下优化问题的解：

    $$f_0(x) = \arg \min_{ \gamma } \sum_{i=1}^N L(y_i, \gamma) $$

2. 对每棵树 $k = 1$ to $M$：

    a. 对每个样本计算负梯度：

    $$ r_{ik} = - \frac{ \partial L(y_i, f(x_i)) }{ \partial f(x_i) } | _{f= f_{k-1}} $$

    b. 以负梯度为目标拟合一个树

    c. 将这个树加入模型，并更新预测值

### Gradient Boosting Trees 实现

以下代码是当损失函数定义为均方损失(mean squared error)时的实现，此时负梯度可以用$g_m = y - f(x)$求得。

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

class MyGbm:
    def __init__(self, tree_num, tree_depth, learn_rate):
        self.tree_num = tree_num
        self.tree_depth = tree_depth
        self.learn_rate = learn_rate

        self.boosting_trees = []
        self.predict0 = 0

    def fit(self, train_X, train_y):
        self.predict0 = train_y.mean()
        prediction = pd.Series(data=self.predict0, index=train_y.index)
        for i in range(1, self.tree_num):
            # assuming loss function is squared error
            # 1. compute negative gradients
            neg_grads = train_y - prediction
            # 2. fit a regression tree
            tree = DecisionTreeRegressor(max_depth=self.tree_depth)
            tree.fit(train_X, neg_grads)
            # 3. update prediction
            self.boosting_trees.append(tree)
            prediction += tree.predict(train_X)*self.learn_rate

    def predict(self, test_X):
        # must be trained
        assert len(self.boosting_trees) > 0
        prediction = pd.Series(data=self.predict0, index=test_X.index)
        for tree in self.boosting_trees:
            prediction += tree.predict(test_X) * self.learn_rate

        return prediction
```

我们可以将其与 sklearn 官方的 `GradientBoostingRegressor` 相比较（通过enable/disable `official`）。选取的数据集是[Black Friday Dataset](https://www.kaggle.com/cerolacia/black-friday-sales-prediction)，比较基准是均方根误差，因为电脑破所以只选择了前 10000 行数据。

```python
official = False
kf = KFold(n_splits=5, shuffle=False, random_state=None)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 10
    if official:
        gbm = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    else:
        gbm = MyGbm(n_estimators, max_depth, learning_rate)
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_test)
    # root mean square error
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    print('rmse:', rmse)
```

官方实现的 GBM：
```console
TRAIN: [2000 2001 2002 ... 9997 9998 9999] TEST: [   0    1    2 ... 1997 1998 1999]
rmse: 3581.9870271000086
TRAIN: [   0    1    2 ... 9997 9998 9999] TEST: [2000 2001 2002 ... 3997 3998 3999]
rmse: 2907.6311604609245
TRAIN: [   0    1    2 ... 9997 9998 9999] TEST: [4000 4001 4002 ... 5997 5998 5999]
rmse: 2944.398603563586
TRAIN: [   0    1    2 ... 9997 9998 9999] TEST: [6000 6001 6002 ... 7997 7998 7999]
rmse: 2999.08249707528
TRAIN: [   0    1    2 ... 7997 7998 7999] TEST: [8000 8001 8002 ... 9997 9998 9999]
rmse: 3416.907798496119
```

自己实现的GBM：
```console
TRAIN: [2000 2001 2002 ... 9997 9998 9999] TEST: [   0    1    2 ... 1997 1998 1999]
rmse: 3607.182565950029
TRAIN: [   0    1    2 ... 9997 9998 9999] TEST: [2000 2001 2002 ... 3997 3998 3999]
rmse: 2907.502318685874
TRAIN: [   0    1    2 ... 9997 9998 9999] TEST: [4000 4001 4002 ... 5997 5998 5999]
rmse: 2941.2904466354057
TRAIN: [   0    1    2 ... 9997 9998 9999] TEST: [6000 6001 6002 ... 7997 7998 7999]
rmse: 3005.347599886747
TRAIN: [   0    1    2 ... 7997 7998 7999] TEST: [8000 8001 8002 ... 9997 9998 9999]
rmse: 3407.221705763073
```

可以看出差距不大，因此这个实现是正确的。


## Reference

1. [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)

2. [Gradient Boosting](https://borgwang.github.io/ml/2019/04/12/gradient-boosting.html)

3. [Black Friday Dataset](https://www.kaggle.com/cerolacia/black-friday-sales-prediction)

4. [Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

5. [Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

6. [Preprocess category features](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features)
