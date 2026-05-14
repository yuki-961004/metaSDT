# %%
# 环境安装�?(如果已经安装成功，以后不需要运行此�?
# %pip install pandas
# %pip install -e ./Python --no-build-isolation

# %%
import pandas
import metaSDT

# %%
# 读取真实数据 (注意：普�?py 文件运行时的当前目录通常是项目根目录，或者该文件所在目录，视你的编辑器配置而定)
# 如果以下路径报错，请尝试改为 "data/data.csv" (如果是以项目根目录运�?
data = pandas.read_csv("data/exp1.csv")

# %%
# ==============================================================================
# 0. 测试 Data Info 数据扫描与轻量级 Pandas 视图
# ==============================================================================
print("\n=== Test 9: Data Info Helper 测试 ===")
# 使用整个完整原始数据�?data，自动匹配列名，并假�?'stim' 为我们要切分�?condition 条件
std_data = metaSDT.info_data(df=data)

# 直接通过字典�?'1' 提取被试 1 的信�?
sub1_info = std_data["subjects"]["1"]

sub1_raw = data.iloc[sub1_info.raw]
if sub1_info.condition:
    first_cond_key = list(sub1_info.condition.keys())[0]
    sub1_condition = data.iloc[sub1_info.condition[first_cond_key]]
else:
    sub1_condition = sub1_raw

# %%
# ==============================================================================
# 测试全新�?matrix_freq 功能
# ==============================================================================
freq_mat = metaSDT.matrix_freq(
    stim=sub1_raw["stim"], resp=sub1_raw["resp"], conf=sub1_raw["conf"]
)

# %%
# ==============================================================================
# 测试全新�?modify_params 功能
# ==============================================================================
# 注意：如果提示找不到模块，请确保已经�?_help 加入到了 Python 包的
# setup.py (�?CMakeLists) 编译列表中，并重新运行了上方�?%pip install -e Python

print("=== Test 1: 扁平 Dict 输入 (默认全部当做 free 参数) ===")
res1 = metaSDT.modify_params({"d": 2.5})
print(res1["d"])  # 预期: [2.5] (注意 C++ 返回的是 vector，所以在 Python 里表现为 list)

print(
    "\n=== Test 2: 降维移动 (使用结构�?Dict，将默认�?free �?c_resp 固定�?fixed) ==="
)
res2 = metaSDT.modify_params({"fixed": {"c_resp": 1.0}})
print(res2["c_resp"])  # 预期: [1.0]

print("\n=== Test 3: Numpy array / List 序列输入 (自适应转化�?C++ vector) ===")
res3 = metaSDT.modify_params({"c_conf": [0.1, 0.5, 0.9, 1.2]})
print(res3["c_conf"])  # 预期: [0.1, 0.5, 0.9, 1.2]

print("\n=== Test 4: 乌龙冲突处理 (同参数出现在多槽，高优先�?free 胜出) ===")
res4 = metaSDT.modify_params(
    {"free": {"rate_decay": 0.8}, "fixed": {"rate_decay": 0.5}}
)
print(res4["rate_decay"])  # 预期: [0.8]

print("\n=== Test 5: 极端情况，传�?None (返回默认参数�? ===")
res5 = metaSDT.modify_params(None)
print(res5["d"])  # 预期: [1.5]

del res1, res2, res3, res4, res5

# %%
# ==============================================================================
# 测试全新�?modify_prior 功能
# ==============================================================================
print("\n=== Test 10: Modify Prior (先验映射测试) ===")
user_priors = {
    "d": {"type": "norm", "mean": 1.5, "sd": 0.5},
    # 给边界数组挂�?Beta 分布，测试一键批量绑�?
    "c_conf": {"type": "beta", "shape1": 2.0, "shape2": 5.0},
}
user_params = {"c_conf": [0.1, 0.5, 0.9]}

prior_res = metaSDT.modify_prior(user_priors, user_params)
print("Compiled Priors (Key is Flat Index):\n", prior_res)

# %%
# ==============================================================================
# 3. 测试 ModelSDT 核心引擎
# ==============================================================================
print("\n=== Test 6: ModelSDT 核心引擎测试 ===")
std_params = metaSDT.modify_params(
    {
        "d": 2.0,
        "c_resp": 0.5,
        "c_conf": [0.1, 0.5, 0.9],  # 使用 [] 输入向量
    }
)

# 直接传入参数，底层会自动读取 c_resp �?c_conf 展开计算
sdt_cdf = metaSDT.model_sdt(std_params)

print(
    f"False Alarm Rates (所有切点的虚报�?:\n {[round(1 - val, 4) for val in sdt_cdf['cdf_noise'][0]]}"
)
print(
    f"Hit Rates (所有切点的击中�?:\n {[round(1 - val, 4) for val in sdt_cdf['cdf_signal'][0]]}"
)

# %%
# ==============================================================================
# 4. 测试 matrix_prob 概率矩阵生成
# ==============================================================================
print("\n=== Test 7: Matrix Prob (理论概率矩阵) ===")
prob_mat = metaSDT.matrix_prob(sdt_cdf["cdf_noise"], sdt_cdf["cdf_signal"], std_params)
print(prob_mat)

# %%
# ==============================================================================
# 5. 测试 matrix_mult 似然矩阵相乘
# ==============================================================================
print("\n=== Test 8: Matrix Mult (似然矩阵相乘) ===")
# 注意：变�?'freq_mat' 是我们在上面步骤跑出来的频数矩阵 DataFrame
metaSDT.matrix_mult(freq_mat, prob_mat, std_params)


# %%
print("\n=== Test 11: Criterion Likelihood ===")
print(metaSDT.criterion_likelihood(freq_mat, prob_mat, std_params))

# %%
print("\n=== Test 12: Criterion Prior (计算一维梯度带先验概率) ===")
# 生成想要评估的当前参数集
current_params = metaSDT.modify_params(
    {"free": {"d": [2.0], "c_conf": [0.2, 0.3, 0.5]}, "fixed": {"c_resp": [0.5]}}
)

print("Log Prior:", metaSDT.criterion_prior(user_priors, current_params))

# %%
print("\n=== Test 13: Criterion Posterior (MCMC/Stan 后验入口测试) ===")
print(
    "Log Posterior:",
    metaSDT.criterion_posterior(freq_mat, user_priors, current_params),
)

# %%
print("\n=== Test 14: Uniform Prior (�?Log-Likelihood 验证) ===")
# 利用均匀分布特性：若上限与下限之差�?1，则先验概率密度�?1，对数先�?log(1)恒为 0�?
# 此时 criterion_posterior 返回的值将 100% 等价于没有任何先验干涉的对数似然 (LogL)�?
log_posterior_unif = metaSDT.criterion_posterior(
    freq_mat,
    {
        "c_conf": {"type": "unif", "min": 0.0, "max": 1.0},
        "d": {"type": "uniform", "lower": 1.5, "upper": 2.5},
    },
    current_params,
)
print(log_posterior_unif)


# %%
# 2. 运行多线MLE 拟合
fit_mle_exp1 = metaSDT.estimate_mle(
    df=pandas.read_csv("data/exp1.csv"),
    colnames={},  # 传入空字典让底层 C++ 自动使用正则去匹配列(stim, resp, conf)
    params={
        "free": {
            "d": [1.5],
            "c_resp": [0.0],
            "c_conf": [0.5, 1.0, 1.5],
        },  # 设定我们要找出的自由参数 (注意 Python 端要求将单值也包裹在列�?[] �?
        "fixed": {"sd_signal": [1.0], "sd_noise": [1.0]},
    },
    model="sdt",
)

# %%
# 2. 运行多线MLE 拟合
fit_mle_exp3 = metaSDT.estimate_mle(
    df=pandas.read_csv("data/exp3.csv"),
    colnames={
        "condition": "FlippedWheel",
        "difficulty": "NoiseLevel_Deg"
    },
    params={
        "free": {
            "d": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "c_resp": [0.0],
            "c_conf": [0.5, 1.0, 1.5],
        },  # 设定我们要找出的自由参数 (注意 Python 端要求将单值也包裹在列�?[] �?
        "fixed": {"sd_signal": [1.0], "sd_noise": [1.0]},
    },
    model="sdt",
)

# %%
# ==============================================================================
# 5. 测试 estimate_map (MAP 估计)
# ==============================================================================
print("\n=== Test 15: estimate_map (Default Priors) ===")
# 先验测试1：留user_priors，测C++ 底层的默SDT 先验能否正常激活并限制极端?
fit_map_exp1 = metaSDT.estimate_map(
    df=pandas.read_csv("data/exp1.csv"),
    colnames={},
    params={
        "free": {
            "d": [1.5],
            "c_resp": [0.0],
            "c_conf": [0.5, 1.0, 1.5],
        },
        "fixed": {"sd_signal": [1.0], "sd_noise": [1.0]},
    },
    model="sdt",
)

# %%
print("\n=== Test 16: estimate_map (Custom Priors) ===")
# 先验测试2：传入自定义 user_priors，测EM-MAP 场景下每次更新先验的过程
fit_map_exp3 = metaSDT.estimate_map(
    df=pandas.read_csv("data/exp3.csv"),
    colnames={
        "condition": "FlippedWheel",
        "difficulty": "NoiseLevel_Deg"
    },
    params={
        "free": {
            "d": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "c_resp": [0.0],
            "c_conf": [0.5, 1.0, 1.5],
        },
        "fixed": {"sd_signal": [1.0], "sd_noise": [1.0]},
    },
    model="sdt",
    user_priors={
        "d": {"type": "norm", "mean": 1.2, "sd": 0.8},
        "c_conf": {"type": "norm", "mean": 0.0, "sd": 1.5}
    }
)


