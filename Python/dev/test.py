# %% 
# 环境安装区 (如果已经安装成功，以后不需要运行此块)
# %pip install pandas 
# %pip install -e Python 

# %%
import pandas
import metaSDT

# %%
# 读取真实数据 (注意：普通 py 文件运行时的当前目录通常是项目根目录，或者该文件所在目录，视你的编辑器配置而定)
# 如果以下路径报错，请尝试改为 "data/data.csv" (如果是以项目根目录运行)
data = pandas.read_csv("data/data.csv")
sub1 = data[data['subj_id'] == 1]

# %%
test = metaSDT.matrix_freq(
    stim = sub1['stim'],
    resp = sub1['resp'],
    conf = sub1['conf']
)

print(test)

# %%
# ==============================================================================
# 测试全新的 modify_params 功能
# ==============================================================================
# 注意：如果提示找不到模块，请确保已经将 _help 加入到了 Python 包的
# setup.py (或 CMakeLists) 编译列表中，并重新运行了上方的 %pip install -e Python

print("=== Test 1: 扁平 Dict 输入 (默认全部当做 free 参数) ===")
res1 = metaSDT.modify_params({"d": 2.5})
print(res1["d"])  # 预期: [2.5] (注意 C++ 返回的是 vector，所以在 Python 里表现为 list)

print("\n=== Test 2: 降维移动 (使用结构化 Dict，将默认在 free 的 c_resp 固定到 fixed) ===")
res2 = metaSDT.modify_params({"fixed": {"c_resp": 1.0}})
print(res2["c_resp"])  # 预期: [1.0]

print("\n=== Test 3: Numpy array / List 序列输入 (自适应转化为 C++ vector) ===")
res3 = metaSDT.modify_params({"c_conf": [0.1, 0.5, 0.9, 1.2]})
print(res3["c_conf"])  # 预期: [0.1, 0.5, 0.9, 1.2]

print("\n=== Test 4: 乌龙冲突处理 (同参数出现在多槽，高优先级 free 胜出) ===")
res4 = metaSDT.modify_params({"free": {"rate_decay": 0.8}, "fixed": {"rate_decay": 0.5}})
print(res4["rate_decay"])  # 预期: [0.8]

print("\n=== Test 5: 极端情况，传入 None (返回默认参数集) ===")
res5 = metaSDT.modify_params(None)
print(res5["d"])       # 预期: [1.5]

# %%
# ==============================================================================
# 3. 测试 ModelSDT 核心引擎
# ==============================================================================
print("\n=== Test 6: ModelSDT 核心引擎测试 ===")
std_params = metaSDT.modify_params({"d": 2.0, "c_resp": 0.5})

# 直接传入模型参数和坐标 x，返回计算结果
res_cdf = metaSDT.model_sdt(std_params, std_params["c_resp"])

print(f"False Alarm Rate (虚报率): {1 - res_cdf['cdf_noise'][0]:.4f}")
print(f"Hit Rate (击中率): {1 - res_cdf['cdf_signal'][0]:.4f}")

# %%
# ==============================================================================
# 4. 测试 matrix_prob 概率矩阵生成
# ==============================================================================
print("\n=== Test 7: Matrix Prob (理论概率矩阵) ===")
prob_mat = metaSDT.matrix_prob(res_cdf['cdf_noise'], res_cdf['cdf_signal'], std_params)
print(prob_mat)