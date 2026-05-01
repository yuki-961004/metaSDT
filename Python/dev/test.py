# %% 
# 环境安装区 (如果已经安装成功，以后不需要运行此块)
# pip install pandas 
# pip install -e Python 

# %%
import pandas
import metaSDT

# 读取真实数据 (注意：普通 py 文件运行时的当前目录通常是项目根目录，或者该文件所在目录，视你的编辑器配置而定)
# 如果以下路径报错，请尝试改为 "data/data.csv" (如果是以项目根目录运行)
data = pandas.read_csv("data/data.csv")
sub1 = data[data['subj_id'] == 1]

# %%
test = metaSDT.matrix_freq(
    stim = sub1['stim'].tolist(),
    resp = sub1['resp'].tolist(),
    conf = sub1['conf'].tolist()
)

print(test)