# LightGBM 参数中的 lambda_l1 和 lambda_l2

LightGBM 的参数大部分意义都能通过名字猜出来，例如 `num_leaves`，`early_stopping` 等。但是有两个常用参数是个异类：

- `lambda_l1`
- `lambda_l2`

顺带与他们相关的还有一个

- `min_gain_to_split`

要调节这些参数需要理解他们背后的含义。但是，lightGBM 到底是如何定义网上似乎没找到把这个说清楚的。官网只是非常简单地提了一下这些是为了正则化。

> 所谓正则化（Regularization）是机器学习中一种常用的技术，其主要目的是控制模型复杂度，减小过拟合。最基本的正则化方法是在原目标（代价）函数 中添加惩罚项，对复杂度高的模型进行“惩罚”。

因此对于 LightGBM 来讲这几个参数肯定是用于控制树节点分裂的。但是似乎也没找到特地针对 LightGBM 如何正则化的介绍，于是我自己看了下源码。

# 1 结论

`lambda_l1`, `lambda_l2` 和 `min_gain_to_split` 都是用于降低模型复杂度，避免过拟合的。

他们作用于每个节点的 gain。gain 在 LightGBM 中用于描述节点上所有样本的训练程度。gain 越小说明训练越充分，分裂价值越低。

gain 的计算方式为：

$$
\frac{[\text{Thresh}(梯度和, \lambda_{L1})]^2}{二阶导数和 + \lambda_{L2}}
$$

对于回归问题，二阶导数为1，其和为节点样本数。


`lambda_l1` 和 `lambda_l2` 都用于加速 gain 的减小。

- `lambda_l1`：设置一个 threshold，gain 小于这个 threshold 直接认为是 0，不再分裂。

- `lambda_l2`：为 gain 的分母（即节点样本数）增加一个常数项，作用于全程，在节点样本数已经很小的时候，能显著减小 gain 避免分裂。

`min_gain_to_split` 的作用可以通过名字猜出。就是如果一个节点的 gain 低于这个数，不再分裂。



# 2 源码解读

Boosting 的每一次迭代都会生成一棵树。在生成树的时候，我们需要寻找每个特征的最佳分割点。

## 2.1 用 gain 来决定是否分割

在 `feature_histogram.hpp` 用于寻找最佳分割点的 `FindBestThresholdSequentially()` 方法中，有如下代码用于决定是否分割：

```cpp
        // current split gain
        double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, meta_->config->lambda_l1,
            meta_->config->lambda_l2, meta_->config->max_delta_step,
            constraints, meta_->monotone_type, meta_->config->path_smooth,
            left_count, right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain <= min_gain_shift) {
          continue;
        }

        // mark as able to be split
        is_splittable_ = true;
```

其中 `min_gain_shift` 由以下函数计算（用到了 `min_gain_to_split`）：
```cpp
  template <bool USE_RAND, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  double BeforeNumercal(double sum_gradient, double sum_hessian, double parent_output, data_size_t num_data,
                        SplitInfo* output, int* rand_threshold) {
    is_splittable_ = false;
    output->monotone_type = meta_->monotone_type;

    double gain_shift = GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
        sum_gradient, sum_hessian, meta_->config->lambda_l1, meta_->config->lambda_l2,
        meta_->config->max_delta_step, meta_->config->path_smooth, num_data, parent_output);
    *rand_threshold = 0;
    if (USE_RAND) {
      if (meta_->num_bin - 2 > 0) {
        *rand_threshold = meta_->rand.NextInt(0, meta_->num_bin - 2);
      }
    }
    return gain_shift + meta_->config->min_gain_to_split;
  }
```

可以看出 `min_gain_shift` 由两部分构成：

- 未分割节点的 gain，由 `GetLeafGain()` 计算
- 配置的 `min_gain_to_split`

与之相比较的 `current_gain` 则是调用了 `GetSplitGains()`：

```cpp
  template <bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  static double GetSplitGains(double sum_left_gradients,
                              double sum_left_hessians,
                              double sum_right_gradients,
                              double sum_right_hessians, double l1, double l2,
                              double max_delta_step,
                              const FeatureConstraint* constraints,
                              int8_t monotone_constraint,
                              double smoothing,
                              data_size_t left_count,
                              data_size_t right_count,
                              double parent_output) {
    if (!USE_MC) {
      return GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(sum_left_gradients,
                                                                sum_left_hessians, l1, l2,
                                                                max_delta_step, smoothing,
                                                                left_count, parent_output) +
             GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(sum_right_gradients,
                                                                sum_right_hessians, l1, l2,
                                                                max_delta_step, smoothing,
                                                                right_count, parent_output);
    }
    // omitted...
  }
```

可以看出，`GetSplitGains()` 实际就是对分割后的两个子节点分别调用`GetLeafGain()`，再将 gains 相加。

所以本质上这个比较的意义是，如果分割后的 gain 小于分割前 gain 外加 `min_gain_to_split`，就不作进一步分割。

现在我们明白了 `min_gain_to_split` 的原理了，`lambda_l1`  和 `lambda_l2` 呢？

## 2.2 如何计算 gain


计算 gain 就需要用到 `lambda_l1` 和 `lambda_l2` 了。

```cpp
  template <bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  static double GetLeafGain(double sum_gradients, double sum_hessians,
                            double l1, double l2, double max_delta_step,
                            double smoothing, data_size_t num_data, double parent_output) {
    if (!USE_MAX_OUTPUT && !USE_SMOOTHING) {
      if (USE_L1) {
        const double sg_l1 = ThresholdL1(sum_gradients, l1);
        return (sg_l1 * sg_l1) / (sum_hessians + l2);
      } else {
        return (sum_gradients * sum_gradients) / (sum_hessians + l2);
      }
    } else {
      double output = CalculateSplittedLeafOutput<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
          sum_gradients, sum_hessians, l1, l2, max_delta_step, smoothing, num_data, parent_output);
      return GetLeafGainGivenOutput<USE_L1>(sum_gradients, sum_hessians, l1, l2, output);
    }
  }
```

gain 的计算方法就是：

$$
\frac{[\text{Thresh}(梯度和, \lambda_{L1})]^2}{二阶导数和 + \lambda_{L2}}
$$

`Thresh` 的算法很简单，就是当梯度和在 $[-\lambda_{L1}, +\lambda_{L1}]$ 区间时，取 0；其余情况不变。
```cpp
  static double ThresholdL1(double s, double l1) {
    const double reg_s = std::max(0.0, std::fabs(s) - l1);
    return Common::Sign(s) * reg_s;
  }
```

二阶导数和需要继续阅读代码。对于 regression 问题，梯度和二阶导数为：
```cpp
  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>(score[i] - label_[i]);
        hessians[i] = 1.0f;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>((score[i] - label_[i]) * weights_[i]);
        hessians[i] = static_cast<score_t>(weights_[i]);
      }
    }
  }
```

这是非常容易理解的，因为在 regression 任务中，LightGBM 采用的目标函数是平方误差：

$$
0.5(y -\hat{y})^2
$$

将其对 y 求导，一阶导数（梯度）就是直接作差，二阶导数是1，__二阶导数和就是该节点样本数量__。

__那么对于 regression 问题，gain 的含义就清楚了。它是用于描述一个节点分裂的“价值”。一个节点中的样本训练越充分，gain 就越小，因为梯度和小__。

同时，`lambda_l1`, `lambda_l2` 的作用也清楚了，__他们都用于加速 gain 减小的过程__。

- `lambda_l1`：设置一个 threshold，gain 小于这个 threshold 直接认为是 0，不再分裂。

- `lambda_l2`：为 gain 的分母（即节点样本数）增加一个常数项，作用于全程，在节点样本数已经很小的时候，能显著减小 gain 避免分裂。
