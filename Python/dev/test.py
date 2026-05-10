# %%
# 环境安装区 (如果已经安装成功，以后不需要运行此块)
# %pip install pandas
# %pip install -e ./Python --config-settings=build-dir="build"

# %%
import pandas
import metaSDT

# %%
# 读取真实数据 (注意：普通 py 文件运行时的当前目录通常是项目根目录，或者该文件所在目录，视你的编辑器配置而定)
# 如果以下路径报错，请尝试改为 "data/data.csv" (如果是以项目根目录运行)
data = pandas.read_csv("data/data.csv")

# %%
# ==============================================================================
# 0. 测试 Data Info 数据扫描与轻量级 Pandas 视图
# ==============================================================================
print("\n=== Test 9: Data Info Helper 测试 ===")
# 使用整个完整原始数据集 data，自动匹配列名，并假定 'stim' 为我们要切分的 condition 条件
std_data = metaSDT.data_info(df=data, condition="stim")

# 直接通过字典键 '1' 提取被试 1 的信息
sub1_info = std_data["subjects"]["1"]

sub1_raw = data.iloc[sub1_info.raw]
sub1_condition = data.iloc[sub1_info.condition[1]]

# %%
# ==============================================================================
# 测试全新的 matrix_freq 功能
# ==============================================================================
freq_mat = metaSDT.matrix_freq(
    stim=sub1_raw["stim"], resp=sub1_raw["resp"], conf=sub1_raw["conf"]
)

# %%
# ==============================================================================
# 测试全新的 modify_params 功能
# ==============================================================================
# 注意：如果提示找不到模块，请确保已经将 _help 加入到了 Python 包的
# setup.py (或 CMakeLists) 编译列表中，并重新运行了上方的 %pip install -e Python

print("=== Test 1: 扁平 Dict 输入 (默认全部当做 free 参数) ===")
res1 = metaSDT.modify_params({"d": 2.5})
print(res1["d"])  # 预期: [2.5] (注意 C++ 返回的是 vector，所以在 Python 里表现为 list)

print(
    "\n=== Test 2: 降维移动 (使用结构化 Dict，将默认在 free 的 c_resp 固定到 fixed) ==="
)
res2 = metaSDT.modify_params({"fixed": {"c_resp": 1.0}})
print(res2["c_resp"])  # 预期: [1.0]

print("\n=== Test 3: Numpy array / List 序列输入 (自适应转化为 C++ vector) ===")
res3 = metaSDT.modify_params({"c_conf": [0.1, 0.5, 0.9, 1.2]})
print(res3["c_conf"])  # 预期: [0.1, 0.5, 0.9, 1.2]

print("\n=== Test 4: 乌龙冲突处理 (同参数出现在多槽，高优先级 free 胜出) ===")
res4 = metaSDT.modify_params(
    {"free": {"rate_decay": 0.8}, "fixed": {"rate_decay": 0.5}}
)
print(res4["rate_decay"])  # 预期: [0.8]

print("\n=== Test 5: 极端情况，传入 None (返回默认参数集) ===")
res5 = metaSDT.modify_params(None)
print(res5["d"])  # 预期: [1.5]

del res1, res2, res3, res4, res5

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

# 直接传入参数，底层会自动读取 c_resp 和 c_conf 展开计算
sdt_cdf = metaSDT.model_sdt(std_params)

print(
    f"False Alarm Rates (所有切点的虚报率):\n {[round(1 - val, 4) for val in sdt_cdf['cdf_noise']]}"
)
print(
    f"Hit Rates (所有切点的击中率):\n {[round(1 - val, 4) for val in sdt_cdf['cdf_signal']]}"
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
# 注意：变量 'freq_mat' 是我们在上面步骤跑出来的频数矩阵 DataFrame
metaSDT.matrix_mult(freq_mat, prob_mat, std_params)


# %%
metaSDT.loss_function(freq_mat, prob_mat, std_params)


# %%
# 2. 运行多线程 MLE 拟合
fit_res = metaSDT.estimate_mle(
    df=pandas.read_csv("data/data.csv"),
    colnames={},  # 传入空字典让底层 C++ 自动使用正则去匹配列名 (如 stim, resp, conf)
    params={
        "free": {
            "d": [1.5],
            "c_resp": [0.0],
            "c_conf": [0.5, 1.0, 1.5],
        },  # 设定我们要找出的自由参数 (注意 Python 端要求将单值也包裹在列表 [] 中)
        "fixed": {"sd_signal": [1.0], "sd_noise": [1.0]},
    },
    model="sdt",
)

print(fit_res.head())
