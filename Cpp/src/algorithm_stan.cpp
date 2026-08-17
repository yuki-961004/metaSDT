#include <metaSDT/algorithm_stan.hpp>

#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/functor/finite_diff_gradient_auto.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

/* ========================================================================== *
 *                              Numeric Helpers                               *
 * ========================================================================== */

// 将概率值 (0, 1) 映射到无界的实数域 (-inf, +inf)。
// 公式为: log(p) - log(1-p) = log(p / (1-p))。
// 例如: logit(0.5) = 0.0, logit(0.9) ≈ 2.197, logit(0.1) ≈ -2.197。
// 在 MCMC 中, 此函数常用于将有上下界的变量解除约束, 从而在全空间进行无障碍采样。
double logit(double probability) {
    return std::log(probability) - std::log1p(-probability);
}

// logit 的反函数 (即 Sigmoid 函数), 将无界实数 (-inf, +inf) 映射回概率值 (0, 1)。
// 公式为: 1 / (1 + exp(-x))。
// 为了避免 x 过大或过小时 exp(x) 计算溢出, 这里根据符号进行了数值稳定的分段处理。
double inv_logit(double value) {
    if (value >= 0.0) {
        const double exp_neg = std::exp(-value);
        return 1.0 / (1.0 + exp_neg);
    }

    const double exp_pos = std::exp(value);
    return exp_pos / (1.0 + exp_pos);
}

// 计算 log(inv_logit(x)), 即 log(1 / (1 + exp(-x)))。
// 通过代数展开为数值稳定的形式, 避免先计算 inv_logit 丢失精度甚至下溢为 0。
double log_inv_logit(double value) {
    if (value >= 0.0) {
        return -std::log1p(std::exp(-value));
    }
    return value - std::log1p(std::exp(value));
}

// 计算 log(1 - inv_logit(x)), 常用于计算带约束参数在进行概率映射时的雅可比(Jacobian)调整项。
// 同样为了数值稳定性进行了符号分段展开。
double log1m_inv_logit(double value) {
    if (value >= 0.0) {
        return -value - std::log1p(std::exp(-value));
    }
    return -std::log1p(std::exp(value));
}

// 将概率值安全地限制在远离 0 和 1 的微小区间内 [eps, 1-eps]。
// 防止极其极端的概率值 (如正好等于 1.0 或 0.0) 在输入给 logit 函数时产生 Inf 或 -Inf。
double clamp_probability(double value) {
    const double eps = 1e-12;
    return std::max(eps, std::min(value, 1.0 - eps));
}

// 检查当前浮点数是否为有限值 (非 NaN 且非 Inf)
bool is_finite(double value) {
    return std::isfinite(value);
}

} // namespace

namespace HMC {
namespace {

// 检查向量中是否所有的元素都是有限的实数 (在 HMC 梯度计算异常时用于拦截)
bool is_finite_vector(const Eigen::VectorXd& values) {
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values(i))) {
            return false;
        }
    }
    return true;
}

// 将对数域的 Metropolis-Hastings 接受率安全地转换为 0.0 到 1.0 之间的概率
double safe_accept_probability(double log_accept_ratio) {
    if (!std::isfinite(log_accept_ratio)) {
        return 0.0;
    }
    if (log_accept_ratio >= 0.0) {
        return 1.0;
    }
    return std::exp(log_accept_ratio);
}

// 从标准正态分布中为动量(Momentum)变量抽取随机样本。
// 这相当于假设 HMC 动力学系统的质量矩阵 (Mass Matrix) 为单位矩阵。
Eigen::VectorXd draw_momentum(
    Eigen::Index n_dim,
    std::mt19937_64& rng
) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd momentum(n_dim);

    // 每个维度独立抽取标准正态动量, 对应单位质量矩阵.
    for (Eigen::Index i = 0; i < n_dim; ++i) {
        momentum(i) = normal(rng);
    }

    return momentum;
}

// 为 MCMC 的初始位置添加轻微的随机扰动 (Jitter)。
// 多链从轻微不同的起点出发, 能够显著降低多条链完全陷入相同轨迹或局部极值的风险。
void jitter_initial(
    Eigen::VectorXd& initial,
    double jitter,
    std::mt19937_64& rng
) {
    if (jitter <= 0.0) {
        return;
    }

    std::normal_distribution<double> normal(0.0, jitter);

    // 多链从轻微不同的起点出发, 降低完全同轨迹的风险.
    for (Eigen::Index i = 0; i < initial.size(); ++i) {
        initial(i) += normal(rng);
    }
}

} // namespace
} // namespace HMC

/* ========================================================================== *
 *                        Stan Posterior Adapter                              *
 * ========================================================================== */

// Adapter 类的构造函数。
// 它封装了计算后验概率所需的全部信息 (包括观测频数、参数名、先验分布等),
// 并持有参数的硬性物理边界 (lower_bounds, upper_bounds)。
// 它是连接项目自有模型和底层 MCMC 采样引擎的"桥梁"。
StanAdapter::Adapter::Adapter(const SubjectFitTask& task)
    : posterior_(
          task.freq.freq_mat,
          task.params.name_free,
          task.params.get_free_sizes(),
          task.params.flat,
          task.prior,
          task.model
      ),
      lower_bounds_(task.params.lower_bounds),
      upper_bounds_(task.params.upper_bounds) {}

// 核心函数: 计算"无约束参数空间"中的对数后验概率 (Log Posterior)。
// 在 HMC/NUTS 算法中, 采样器像在平原上漫游一样在一个无界空间 (-inf, +inf) 中运动 (比如丢出一个 z = 15.0),
// 此函数负责把这漫游的坐标转换为实际模型能理解的分数:
// 1. 将 z 映射回实际有界的模型参数 (比如 [0, 1] 之间的 0.999)。
// 2. 用真正的参数去计算模型的对数似然 (Log Likelihood) 和先验。
// 3. 加上"雅可比行列式对数" (log_jacobian) 以补偿坐标映射带来的概率密度形变。
// 
// [通俗比喻]: 就像把地球仪 (三维球面) 摊平画在世界地图 (二维平面) 上, 靠近极点的地方面积会被拉伸。
// 雅可比行列式就是用来告诉系统: "这里被拉伸了多少倍，请在计算概率密度时把误差扣除掉"。
double StanAdapter::Adapter::criterion(
    const Eigen::VectorXd& unconstrained
) const {
    ++n_evals_;

    double log_jacobian = 0.0;
    const Eigen::VectorXd constrained = constrain(
        unconstrained,
        log_jacobian
    );

    // 如果参数转换失败, 返回负无穷, 让 HMC 自动拒绝该点.
    if (!is_finite(log_jacobian)) {
        return -std::numeric_limits<double>::infinity();
    }

    const double log_posterior = posterior_.operator()<double>(constrained);

    // CriterionPosterior 的值必须有限, 否则该 proposal 无效.
    if (!is_finite(log_posterior)) {
        return -std::numeric_limits<double>::infinity();
    }

    return log_posterior + log_jacobian;
}

// 计算当前位置的对数概率和对应的"梯度" (Gradient)。
// [原理]: HMC 模拟了物理学中的滑板滑下山坡。要知道滑板往哪边加速, 就必须知道当前脚下山坡的倾斜度(坡度), 这就是梯度。
// [实现]: 这里使用 Stan Math 提供的有限差分法 (finite_diff_gradient_auto), 
// 它会尝试把每个维度的参数微微挪动一点点 (比如 +0.0001), 看看概率变了多少, 从而推算出偏导数。
// 它将计算出的概率存入 log_prob, 坡度存入 gradient 向量。
void StanAdapter::Adapter::gradient(
    const Eigen::VectorXd& unconstrained,
    double& log_prob,
    Eigen::VectorXd& gradient
) const {
    gradient.resize(unconstrained.size());

    // Stan Math 在这里像坡度尺, 只负责量出 criterion 的局部坡度.
    auto criterion_fn = [this](const Eigen::VectorXd& value) {
        return this->criterion(value);
    };
    Eigen::VectorXd work = unconstrained;
    stan::math::finite_diff_gradient_auto(
        criterion_fn,
        work,
        log_prob,
        gradient
    );
}

// constrain 函数: 将无界空间变量 (z) 压缩、映射回受限的模型参数空间 (x)。
// 同时累加变换过程中产生的 log_jacobian 调整项。
// [映射举例]:
// - 双边界 [L, U]: 假设边界是 [0, 10]。采样器给出一个 z = 0.0。inv_logit(0) 是 0.5 (即 50% 进度)。映射结果 x = 0 + 10 * 0.5 = 5.0。
// - 仅下界 [L, inf): 假设下界是 2.0。采样器给出一个 z = 0.0。映射结果 x = 2.0 + exp(0) = 2.0 + 1.0 = 3.0。
// - 仅上界 (-inf, U]: 映射结果 x = U - exp(z)。
Eigen::VectorXd StanAdapter::Adapter::constrain(
    const Eigen::VectorXd& unconstrained,
    double& log_jacobian
) const {
    const Eigen::Index n_dim = unconstrained.size();
    Eigen::VectorXd constrained(n_dim);
    log_jacobian = 0.0;

    for (Eigen::Index i = 0; i < n_dim; ++i) {
        const double z = unconstrained(i);
        const double lower = lower_bounds_[static_cast<size_t>(i)];
        const double upper = upper_bounds_[static_cast<size_t>(i)];
        const bool has_lower = is_finite(lower);
        const bool has_upper = is_finite(upper);

        if (has_lower && has_upper) {
            const double width = upper - lower;

            // 上下界必须形成有效区间, 否则目标函数没有定义.
            if (width <= 0.0) {
                log_jacobian = -std::numeric_limits<double>::infinity();
                constrained(i) = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            const double probability = inv_logit(z);
            constrained(i) = lower + width * probability;
            log_jacobian += std::log(width);
            log_jacobian += log_inv_logit(z);
            log_jacobian += log1m_inv_logit(z);
        } else if (has_lower) {
            constrained(i) = lower + std::exp(z);
            log_jacobian += z;
        } else if (has_upper) {
            constrained(i) = upper - std::exp(z);
            log_jacobian += z;
        } else {
            constrained(i) = z;
        }
    }

    return constrained;
}

// unconstrain 函数: constrain 的逆运算。
// 将用户提供的、在物理边界内的初始值 (x), 反向撕扯拉伸到无约束的采样器空间 (z)。
// [举例]: 如果参数双边界是 [0, 10], 初始值是 5.0 (正好在正中间)。
// 系统会算出它占了区间的 50% (0.5)。反向拉伸: logit(0.5) = 0.0。
// 于是采样器就会从 0.0 这个坐标开始它第一步的 MCMC 漫游。
Eigen::VectorXd StanAdapter::Adapter::unconstrain(
    const std::vector<double>& constrained
) const {
    Eigen::VectorXd unconstrained(
        static_cast<Eigen::Index>(constrained.size())
    );

    for (size_t i = 0; i < constrained.size(); ++i) {
        const double value = constrained[i];
        const double lower = lower_bounds_[i];
        const double upper = upper_bounds_[i];
        const bool has_lower = is_finite(lower);
        const bool has_upper = is_finite(upper);

        if (has_lower && has_upper) {
            const double width = upper - lower;
            if (width <= 0.0) {
                throw std::invalid_argument(
                    "Stan transform received invalid parameter bounds."
                );
            }

            const double probability = clamp_probability(
                (value - lower) / width
            );
            unconstrained(static_cast<Eigen::Index>(i)) =
                logit(probability);
        } else if (has_lower) {
            unconstrained(static_cast<Eigen::Index>(i)) =
                std::log(std::max(value - lower, 1e-12));
        } else if (has_upper) {
            unconstrained(static_cast<Eigen::Index>(i)) =
                std::log(std::max(upper - value, 1e-12));
        } else {
            unconstrained(static_cast<Eigen::Index>(i)) = value;
        }
    }

    return unconstrained;
}

// 辅助函数: 仅进行 constrain 约束转换, 返回标准 C++ 向量 (std::vector), 并且直接丢弃雅可比调整项。
// [用途]: 当 MCMC 链接受了某一次位置跃迁后, 我们需要把当前坐标"翻译"成人能看懂的真实模型参数并记录下来 (即保存 Draw/Sample)。
// 此时我们只关心参数值本身, 不再需要算概率密度, 所以忽略 log_jacobian 节省计算。
std::vector<double> StanAdapter::Adapter::constrain_to_vector(
    const Eigen::VectorXd& unconstrained
) const {
    double log_jacobian = 0.0;
    const Eigen::VectorXd constrained = constrain(
        unconstrained,
        log_jacobian
    );

    std::vector<double> out(static_cast<size_t>(constrained.size()));
    for (Eigen::Index i = 0; i < constrained.size(); ++i) {
        out[static_cast<size_t>(i)] = constrained(i);
    }

    return out;
}

// 计数器: 返回目标函数 (似然/后验概率) 到底被调用计算了多少次。
// 用于外层封装进行算法性能诊断。
int StanAdapter::Adapter::n_evals() const {
    return n_evals_;
}

/* ========================================================================== *
 *                       Initial Value Bound Handling                         *
 * ========================================================================== */

// 将用户提供的初始参数点安全地推入有效边界内部。
// [原理]: 纯数学上的概率边界 (如概率正好等于 1.0 或 0.0) 会导致对数雅可比计算出无穷大或 NaN。
// MCMC 采样器极其脆弱，初始点如果在这些致命边界上，整个链会直接崩溃。
// [实现]: 引入一个微小的偏移量 epsilon。
// - 如果卡在下界, 就把它往上推一点点 (lower + eps)。
// - 如果卡在上界, 就把它往下推一点点 (upper - eps)。
void StanAdapter::sanitize_initial_point(
    std::vector<double>& initial,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double epsilon
) {
    for (size_t i = 0; i < initial.size(); ++i) {
        const bool has_lower = is_finite(lower_bounds[i]);
        const bool has_upper = is_finite(upper_bounds[i]);

        if (has_lower && initial[i] <= lower_bounds[i]) {
            // 有下界时, 初始值必须推入可行域内部.
            initial[i] = lower_bounds[i] + epsilon;
        }

        if (has_upper && initial[i] >= upper_bounds[i]) {
            // 有上界时, 初始值必须推入可行域内部.
            initial[i] = upper_bounds[i] - epsilon;
        }

        if (has_lower && has_upper) {
            const double midpoint =
                0.5 * (lower_bounds[i] + upper_bounds[i]);

            // 如果边界太窄, 修正后仍不合法, 就回到区间中点.
            if (initial[i] <= lower_bounds[i] ||
                initial[i] >= upper_bounds[i]) {
                initial[i] = midpoint;
            }
        }
    }
}

namespace NUTS {
namespace {

/* ========================================================================== *
 *                             NUTS Tree Logic                                *
 * ========================================================================== */

bool is_finite_vector(const Eigen::VectorXd& values) {
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values(i))) {
            return false;
        }
    }
    return true;
}

double safe_accept_probability(double log_accept_ratio) {
    if (!std::isfinite(log_accept_ratio)) {
        return 0.0;
    }
    if (log_accept_ratio >= 0.0) {
        return 1.0;
    }
    return std::exp(log_accept_ratio);
}

Eigen::VectorXd draw_momentum(
    Eigen::Index n_dim,
    std::mt19937_64& rng
) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd momentum(n_dim);

    // 每个维度独立抽取标准正态动量, 对应单位质量矩阵.
    for (Eigen::Index i = 0; i < n_dim; ++i) {
        momentum(i) = normal(rng);
    }

    return momentum;
}

void jitter_initial(
    Eigen::VectorXd& initial,
    double jitter,
    std::mt19937_64& rng
) {
    if (jitter <= 0.0) {
        return;
    }

    std::normal_distribution<double> normal(0.0, jitter);

    // 多链从轻微不同的起点出发, 降低完全同轨迹的风险.
    for (Eigen::Index i = 0; i < initial.size(); ++i) {
        initial(i) += normal(rng);
    }
}

// 物理系统中的"状态" (State) 快照。
// [比喻]: 这就像是摄像机拍下了滑板运动员在某一帧的完整信息:
// 位置 (position，对应参数值)、速度 (momentum)、当前脚下山坡的坡度 (gradient)，
// 以及当前所处位置的海拔高度 (log_prob, 对数后验概率)。
struct State {
    Eigen::VectorXd position;
    Eigen::VectorXd momentum;
    Eigen::VectorXd gradient;
    double log_prob = -std::numeric_limits<double>::infinity();
    bool valid = false;
};

// 递归构建 NUTS 二叉树时返回的"子树结果" (TreeResult)。
// NUTS 算法不是盲目向前走，而是通过向两端翻倍扩展二叉树来探索轨迹。
// 它记录了这段轨迹的: 
// - 两端边界 (left, right) 用于后续的掉头检测;
// - 沿途挑选出的最佳候选点 (candidate);
struct TreeResult {
    State left;
    State right;
    State candidate;

    int n_valid = 0;
    int accept_count = 0;

    double accept_sum = 0.0;

    bool valid_candidate = false;
    bool continue_tree = false;
};

// U 型转弯 (U-Turn) 检测。这也是 NUTS (No-U-Turn Sampler) 名字的由来！
// [原理]: 判断从轨迹起点(left)到终点(right)的位移向量，是否开始与动量(速度)方向背道而驰。
// 通过计算点乘 (dot product) 来判断夹角: 如果点乘 < 0，说明夹角大于90度，即开始回头了。
// [比喻]: 如果你滑滑板滑过了谷底，动能耗尽开始向后倒退，这就形成了一个 U-Turn。
// 此时继续模拟只是在原地绕圈浪费算力，所以一旦触发 U-Turn，NUTS 就会停止树的生长。
bool no_u_turn(
    const State& left,
    const State& right
) {
    const Eigen::VectorXd delta = right.position - left.position;
    return delta.dot(left.momentum) >= 0.0 &&
        delta.dot(right.momentum) >= 0.0;
}

// 蛙跳积分器 (Leapfrog Integrator)。
// 它是 HMC 模拟物理运动的核心引擎，具有"时间可逆"和"保体积"的优良数学特性。
// [步伐拆解]:
// 1. 半步动量加速: momentum += 0.5 * step * gradient
// 2. 整步位置滑行: position += 1.0 * step * momentum
// 3. 重新测量坡度: 算新的 gradient
// 4. 再半步动量加速: momentum += 0.5 * step * gradient
State leapfrog(
    const StanAdapter::Adapter& adapter,
    const State& start,
    double step_size,
    int direction
) {
    State next;
    const double signed_step =
        static_cast<double>(direction) * step_size;

    // 先走半步动量, 再走整步位置, 最后用新梯度补齐半步.
    Eigen::VectorXd momentum =
        start.momentum + 0.5 * signed_step * start.gradient;
    next.position = start.position + signed_step * momentum;

    adapter.gradient(
        next.position,
        next.log_prob,
        next.gradient
    );

    if (!std::isfinite(next.log_prob) ||
        !is_finite_vector(next.gradient)) {
        next.valid = false;
        return next;
    }

    next.momentum = momentum + 0.5 * signed_step * next.gradient;
    next.valid = is_finite_vector(next.position) &&
        is_finite_vector(next.momentum);
    return next;
}

// NUTS 最复杂的递归核心: 轨迹树的生长 (Build Tree)。
// 每次调用都会将轨迹在时间上翻倍 (增加 2^depth 个节点)。
// [过程]:
// - 当 depth == 0 时 (叶子节点): 只执行一次 leapfrog 迈出一步。
// - 当 depth > 0 时: 先长出半棵树 (长度 L), 如果没掉头，再从边缘长出后半棵树 (长度 L)，合并成 2L 的新树。
// [公平采样]: 在合并时，会根据两棵子树中"有效点"的数量 (n_valid) 作为权重，按比例随机决定是否用新树的候选点替换旧树的候选点。这保证了轨迹上的每个有效点被选中的概率是均等的。
TreeResult build_tree(
    const StanAdapter::Adapter& adapter,
    const State& state,
    int direction,
    int depth,
    double log_slice,
    double initial_joint,
    double step_size,
    double max_delta_energy,
    std::mt19937_64& rng
) {
    if (depth == 0) {
        TreeResult out;
        const State next = leapfrog(
            adapter,
            state,
            step_size,
            direction
        );
        out.left = next;
        out.right = next;
        out.candidate = next;
        out.accept_count = 1;

        if (!next.valid) {
            out.continue_tree = false;
            return out;
        }

        const double joint =
            next.log_prob - 0.5 * next.momentum.squaredNorm();
        out.n_valid = (log_slice <= joint) ? 1 : 0;
        out.valid_candidate = out.n_valid > 0;
        out.continue_tree = (log_slice - max_delta_energy) < joint;
        out.accept_sum = safe_accept_probability(
            joint - initial_joint
        );
        return out;
    }

    TreeResult first = build_tree(
        adapter,
        state,
        direction,
        depth - 1,
        log_slice,
        initial_joint,
        step_size,
        max_delta_energy,
        rng
    );

    if (!first.continue_tree) {
        return first;
    }

    const State& edge = (direction == -1) ? first.left : first.right;
    TreeResult second = build_tree(
        adapter,
        edge,
        direction,
        depth - 1,
        log_slice,
        initial_joint,
        step_size,
        max_delta_energy,
        rng
    );

    TreeResult out;
    if (direction == -1) {
        out.left = second.left;
        out.right = first.right;
    } else {
        out.left = first.left;
        out.right = second.right;
    }

    out.candidate = first.candidate;
    out.valid_candidate = first.valid_candidate;

    const int combined_valid = first.n_valid + second.n_valid;
    if (second.valid_candidate && combined_valid > 0) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double choose_second =
            static_cast<double>(second.n_valid) /
            static_cast<double>(combined_valid);

        // 用子树有效点数量作为权重, 保持从切片集合中近似均匀选择.
        if (!out.valid_candidate || uniform(rng) < choose_second) {
            out.candidate = second.candidate;
            out.valid_candidate = true;
        }
    }

    out.n_valid = combined_valid;
    out.accept_count = first.accept_count + second.accept_count;
    out.accept_sum = first.accept_sum + second.accept_sum;
    out.continue_tree = first.continue_tree &&
        second.continue_tree &&
        no_u_turn(out.left, out.right);
    return out;
}

} // namespace
} // namespace NUTS

/* ========================================================================== *
 *                            HMC Sampler Entry                               *
 * ========================================================================== */

// 传统的静态哈密顿蒙特卡洛 (Static HMC) 单链采样循环。
// [特点]: 它每一步必须走固定的步数 (`leapfrog_steps`)。
// 缺点是如果你把步数设得太大，滑板会在谷底来回晃荡浪费时间 (U-Turn)；
// 设得太小，又会导致粒子移动太慢 (Random Walk 行为)。
HMCSamplerResult HMC::run_chain(
    const StanAdapter::Adapter& adapter,
    const Eigen::VectorXd& initial_unconstrained,
    const StanControl& control,
    int chain_id
) {
    HMCSamplerResult out;
    out.final_step_size = control.step_size;

    const Eigen::Index n_dim = initial_unconstrained.size();
    if (n_dim <= 0) {
        out.status = -1;
        out.result_message = "MCMC requires at least one free parameter.";
        out.stop_reason = "empty_parameter";
        return out;
    }

    const unsigned long long chain_seed =
        static_cast<unsigned long long>(control.seed) +
        static_cast<unsigned long long>(chain_id) * 104729ULL;
    std::mt19937_64 rng(chain_seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    Eigen::VectorXd current = initial_unconstrained;
    jitter_initial(current, control.initial_jitter, rng);

    double current_log_prob = 0.0;
    Eigen::VectorXd current_gradient(n_dim);
    adapter.gradient(
        current,
        current_log_prob,
        current_gradient
    );

    if (!std::isfinite(current_log_prob) ||
        !is_finite_vector(current_gradient)) {
        out.status = -1;
        out.result_message = "Initial MCMC state has invalid log density.";
        out.stop_reason = "invalid_initial_state";
        return out;
    }

    double step_size = control.step_size;
    const int total_iterations =
        control.warmup + control.samples * control.thin;
    out.draws.reserve(static_cast<size_t>(control.samples));
    out.log_prob.reserve(static_cast<size_t>(control.samples));

    for (int iter = 0; iter < total_iterations; ++iter) {
        const Eigen::VectorXd initial_momentum =
            draw_momentum(n_dim, rng);
        Eigen::VectorXd proposed = current;
        Eigen::VectorXd proposed_momentum = initial_momentum;
        Eigen::VectorXd proposed_gradient = current_gradient;
        double proposed_log_prob = current_log_prob;
        bool valid_proposal = true;

        // leapfrog 使用 log posterior 梯度, 所以动量沿上升方向更新.
        proposed_momentum += 0.5 * step_size * proposed_gradient;

        for (int leap = 0; leap < control.leapfrog_steps; ++leap) {
            proposed += step_size * proposed_momentum;

            adapter.gradient(
                proposed,
                proposed_log_prob,
                proposed_gradient
            );

            if (!std::isfinite(proposed_log_prob) ||
                !is_finite_vector(proposed_gradient)) {
                valid_proposal = false;
                break;
            }

            // 最后一步只做半步动量更新, 这样 leapfrog 保持对称.
            if (leap + 1 < control.leapfrog_steps) {
                proposed_momentum += step_size * proposed_gradient;
            }
        }

        double accept_probability = 0.0;
        if (valid_proposal) {
            proposed_momentum += 0.5 * step_size * proposed_gradient;
            // 反转动量以保证数学上的完全可逆性 (虽然在计算能量时由于平方会抵消)。
            proposed_momentum = -proposed_momentum;

            const double current_energy =
                -current_log_prob +
                0.5 * initial_momentum.squaredNorm();
            const double proposed_energy =
                -proposed_log_prob +
                0.5 * proposed_momentum.squaredNorm();
            const double log_accept_ratio =
                current_energy - proposed_energy;
            accept_probability =
                safe_accept_probability(log_accept_ratio);

            if (std::log(uniform(rng)) < log_accept_ratio) {
                current = proposed;
                current_log_prob = proposed_log_prob;
                current_gradient = proposed_gradient;
                out.n_accept += 1;
            }
        }

        out.n_proposals += 1;

        // Dual-Averaging 步长自适应阶段。
        // 在 Warmup (预热) 期间，如果接受率偏低，就缩小步长让脚步变扎实；
        // 如果接受率偏高，就放大步长让探索更快。
        if (iter < control.warmup && control.adapt_step_size) {
            const double adapt_rate =
                1.0 / std::sqrt(static_cast<double>(iter + 1));
            const double log_step =
                std::log(step_size) +
                adapt_rate * (accept_probability - control.target_accept);
            step_size = std::exp(log_step);
            step_size = std::max(
                control.min_step_size,
                std::min(step_size, control.max_step_size)
            );
        }

        if (iter >= control.warmup) {
            const int sampling_iter = iter - control.warmup;

            // thin 鎺у埗淇濈暀闂撮殧, 鍏朵綑杩唬鍙敤浜庢帹杩?Markov chain
            if (sampling_iter % control.thin == 0) {
                out.draws.push_back(adapter.constrain_to_vector(current));
                out.log_prob.push_back(current_log_prob);
            }
        }
    }

    out.final_step_size = step_size;
    if (out.n_proposals > 0) {
        out.accept_rate =
            static_cast<double>(out.n_accept) /
            static_cast<double>(out.n_proposals);
    }

    if (static_cast<int>(out.draws.size()) == control.samples) {
        out.status = 1;
        out.result_message = "HMC sampling finished.";
        out.stop_reason = "complete";
    } else {
        out.status = -1;
        out.result_message = "HMC produced fewer draws than needed.";
        out.stop_reason = "insufficient_draws";
    }

    return out;
}

/* ========================================================================== *
 *                            NUTS Sampler Entry                              *
 * ========================================================================== */

// No-U-Turn Sampler (NUTS) 单链采样循环。
// [特点]: 这是现代概率编程语言 (如 Stan, PyMC3) 的默认顶级引擎。
// 相比于上面固定步数的 HMC，它极其聪明: 它通过 `build_tree` 动态地指数级扩展轨迹，
// 直到探测到 U-Turn 掉头行为才停下。
// 这样既避免了原地绕圈浪费算力，又免去了用户手动调参的痛苦。
HMCSamplerResult NUTS::run_chain(
    const StanAdapter::Adapter& adapter,
    const Eigen::VectorXd& initial_unconstrained,
    const StanControl& control,
    int chain_id
) {
    HMCSamplerResult out;
    out.final_step_size = control.step_size;

    const Eigen::Index n_dim = initial_unconstrained.size();
    if (n_dim <= 0) {
        out.status = -1;
        out.result_message = "NUTS requires at least one free parameter.";
        out.stop_reason = "empty_parameter";
        return out;
    }

    const unsigned long long chain_seed =
        static_cast<unsigned long long>(control.seed) +
        static_cast<unsigned long long>(chain_id) * 104729ULL;
    std::mt19937_64 rng(chain_seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::exponential_distribution<double> exponential(1.0);

    State current;
    current.position = initial_unconstrained;
    jitter_initial(current.position, control.initial_jitter, rng);
    adapter.gradient(
        current.position,
        current.log_prob,
        current.gradient
    );
    current.valid = std::isfinite(current.log_prob) &&
        is_finite_vector(current.gradient);

    if (!current.valid) {
        out.status = -1;
        out.result_message = "Initial NUTS state has invalid log density.";
        out.stop_reason = "invalid_initial_state";
        return out;
    }

    double step_size = control.step_size;
    double accept_rate_sum = 0.0;
    const int total_iterations =
        control.warmup + control.samples * control.thin;
    out.draws.reserve(static_cast<size_t>(control.samples));
    out.log_prob.reserve(static_cast<size_t>(control.samples));

    for (int iter = 0; iter < total_iterations; ++iter) {
        current.momentum = draw_momentum(n_dim, rng);
        const double initial_joint =
            current.log_prob - 0.5 * current.momentum.squaredNorm();
        const double log_slice = initial_joint - exponential(rng);

        State left = current;
        State right = current;
        State proposal = current;

        int n_valid = 1;
        int depth = 0;
        bool keep_sampling = true;
        double accept_sum = 0.0;
        int accept_count = 0;

        // 动态翻倍扩展轨迹，直到触发 U-Turn (keep_sampling 变为 false)
        // 或者达到了最大树深度限制 (防止在极其平坦的后验空间中无限生长耗尽内存)。
        while (keep_sampling && depth < control.max_tree_depth) {
            // 抛硬币决定这棵树是向着未来(1)长，还是向着过去(-1)长。
            const int direction = (uniform(rng) < 0.5) ? -1 : 1;
            TreeResult tree;

            if (direction == -1) {
                tree = build_tree(
                    adapter,
                    left,
                    direction,
                    depth,
                    log_slice,
                    initial_joint,
                    step_size,
                    control.max_delta_energy,
                    rng
                );
                left = tree.left;
            } else {
                tree = build_tree(
                    adapter,
                    right,
                    direction,
                    depth,
                    log_slice,
                    initial_joint,
                    step_size,
                    control.max_delta_energy,
                    rng
                );
                right = tree.right;
            }

            if (tree.valid_candidate && tree.n_valid > 0) {
                const int combined_valid = n_valid + tree.n_valid;
                const double choose_tree =
                    static_cast<double>(tree.n_valid) /
                    static_cast<double>(combined_valid);

                // 按有效候选数量合并旧树与新树, 避免偏向较早节点.
                if (uniform(rng) < choose_tree) {
                    proposal = tree.candidate;
                }
                n_valid = combined_valid;
            }

            accept_sum += tree.accept_sum;
            accept_count += tree.accept_count;
            keep_sampling = tree.continue_tree &&
                no_u_turn(left, right);
            ++depth;
        }

        if (proposal.valid) {
            current.position = proposal.position;
            current.log_prob = proposal.log_prob;
            current.gradient = proposal.gradient;
        }

        const double accept_probability = (accept_count > 0)
            ? accept_sum / static_cast<double>(accept_count)
            : 0.0;
        accept_rate_sum += accept_probability;
        out.n_proposals += 1;

        if (iter < control.warmup && control.adapt_step_size) {
            const double adapt_rate =
                1.0 / std::sqrt(static_cast<double>(iter + 1));
            const double log_step =
                std::log(step_size) +
                adapt_rate * (accept_probability - control.target_accept);
            step_size = std::exp(log_step);
            step_size = std::max(
                control.min_step_size,
                std::min(step_size, control.max_step_size)
            );
        }

        if (iter >= control.warmup) {
            const int sampling_iter = iter - control.warmup;
            if (sampling_iter % control.thin == 0) {
                out.draws.push_back(
                    adapter.constrain_to_vector(current.position)
                );
                out.log_prob.push_back(current.log_prob);
            }
        }
    }

    out.final_step_size = step_size;
    if (out.n_proposals > 0) {
        out.accept_rate =
            accept_rate_sum / static_cast<double>(out.n_proposals);
    }

    if (static_cast<int>(out.draws.size()) == control.samples) {
        out.status = 1;
        out.result_message = "NUTS sampling finished.";
        out.stop_reason = "complete";
    } else {
        out.status = -1;
        out.result_message = "NUTS produced fewer draws than needed.";
        out.stop_reason = "insufficient_draws";
    }

    return out;
}
