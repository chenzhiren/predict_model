import pandas as pd
import numpy as np
# 查看缺失值情况
def mis_value(data):
    data = data.replace('', np.nan)
    mis_val=data.isnull().sum()  # 计算出每一列有多少个缺失值,因为True=1,False=0
    mis_val_per=data.isnull().sum()/len(data)*100 # 计算每一列缺失值在该列占比多少
    mis_val_table=pd.concat([mis_val,mis_val_per],axis=1) #将两个Series横向拼接
    mis_val_table=mis_val_table.rename(columns={0:'缺失值个数',1:'缺失比例'})
    mis_pai=mis_val_table.sort_values(by='缺失比例',ascending=False) # 按缺失值比例进行排序
    return mis_pai