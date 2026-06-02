import pandas as pd

# 读取CSV文件
# df = pd.read_csv('/root/lanyun-tmp/dataset_ddsm/newtest1.csv')
df = pd.read_csv('/root/lanyun-tmp/dataset_ddsm/newvalid1.csv')

# 对'patient_id'和'laterality'进行分组，并检查每个组中'cancer'的唯一值
groups = df.groupby(['patient_id', 'laterality'])['cancer'].nunique()

# 找出'cancer'有多个唯一值的组
inconsistent_groups = groups[groups > 1]

# 打印出这些组
print(inconsistent_groups)