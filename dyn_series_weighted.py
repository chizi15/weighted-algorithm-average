import numpy as np
import pandas as pd
import random


def dyn_seri_weighted(seri, type=None, w=None, initial=1, r=2, d=1/4):
    """
    传入一维数组seri，可以是series,array,list,tuple；若type='geometric'或'arithmetic'，且输入了w，则w不起作用；若不输入权重，则根据seri的长度动态计算基于几何级数或算数级数再作归一化的权重，再做算术平均；
    也可人为输入权重做算术平均；若不输入type和w，则进行简单算数平均；因为使用np.dot，则seri索引越小，权重越大；将seri各点与权重相乘再相加，得到一个最终点。
    :param seri: 需要进行加权变成一个点的一维数组
    :param type: 采用几何级数或算数级数进行加权，或人为指定权重，或默认权重相等，type = 'geometric'或'arithmetic'或None；若type='geometric'或'arithmetic'，且输入了w，则w不起作用。
    :param w: 一维的权重系数，可以是series,array,list,tuple；若手动输入，其长度必须和一维数组seri（即序列点数）相等
    :param r: 指定几何级数分母的公比；r越大，权重衰减越快
    :param d: 指定算数级数分母的公差；d越大，权重衰减越快
    :param initial: 指定算数级数分母的初始值；initial越小，初始点所占权重越大；对于几何级数，initial取值对权重无影响，因其可作为分母的公因子提出，与分子上的initial抵消
    :return: seri各点与权重w相乘再相加，返回的一个加权后的最终点；以及seri从左至右的各点权重。
    """
    if type not in ['geometric', 'arithmetic', None]:
        raise Exception('type must be one of geometric, arithmetic or None')
    if type is not None:
        w = list()
        if type == 'geometric':
            for i in range(len(seri)):
                w.append(initial * (1 / r) ** i)  # 生成首项是initial，公比是(1/r)的几何级数作权重
        else:
            for i in range(len(seri)):
                w.append(1 / (initial + d * i))  # 生成首项是initial，公差是d的算术级数，再做倒数作为权重
        w = np.array(w) / sum(w)
    elif (type is None) and (w is None):
        w = np.ones(len(seri)) / sum(np.ones(len(seri)))  # 生成均等权重
    elif (type is None) and (w is not None) and (len(w) == len(seri)):
        w = np.array(w) / sum(w)  # 自定义权重
    else:
        raise Exception('手动输入的权重长度必须和一维数组长度（即序列点数）相等')
    if abs(sum(w)-1) > 0.001:
        raise Exception('weights are not useable')
    return round(np.dot(np.array(seri), w), 4), w


def dyn_df_weighted(df, type=None, w=None, initial=1*2, r=2, d=1/2):
    """
    传入二维数组df；若type='geometric'或'arithmetic'，且输入了w，则w不起作用；若不输入权重，则根据df的列数动态计算基于几何级数或算数级数再作归一化的权重，再做算术平均；
    也可人为输入权重做算术平均；若不输入type和w，则进行简单算数平均；因为使用np.matmul，则df.columns的索引越小，权重越大；将df的各列与权重相乘再相加，得到一条最终的序列。
    :param df: 需要进行加权变成一条序列的二维数组，df的每列代表一条需要进行加权的序列
    :param type: 采用几何级数或算数级数进行加权，或人为指定权重，或默认权重相等，type = 'geometric'或'arithmetic'或None；若type='geometric'或'arithmetic'，且输入了w，则w不起作用。
    :param w: 一维的权重系数，可以是series,array,list,tuple；若手动输入，其长度必须和一维数组seri（即序列点数）相等
    :param r: 指定几何级数分母的公比；r越大，权重衰减越快
    :param d: 指定算数级数分母的公差；d越大，权重衰减越快
    :param initial: 指定算数级数分母的初始值；initial越小，初始点所占权重越大；对于几何级数，initial取值对权重无影响，因其可作为分母的公因子提出，与分子上的initial抵消
    :return: df各列与权重w相乘再相加，返回一条最终的序列；以及seri从左至右的各点权重。
    """
    if type not in ['geometric', 'arithmetic', None]:
        raise Exception('type must be one of geometric, arithmetic or None')
    if type is not None:
        w = list()
        if type == 'geometric':
            for i in range(len(df.columns)):
                w.append(initial * (1 / r) ** i)  # 生成首项是initial，公比是(1/r)的几何级数作权重
        else:
            for i in range(len(df.columns)):
                w.append(1 / (initial + d * i))  # 生成首项是initial，公差是d的算术级数，再做倒数作为权重
        w = np.array(w) / sum(w)
    elif (type is None) and (w is None):
        w = np.ones(len(df.columns)) / sum(np.ones(len(df.columns)))  # 生成均等权重
    elif (type is None) and (w is not None) and (len(w) == len(df.columns)):
        w = np.array(w) / sum(w)  # 自定义权重
    else:
        raise Exception('手动输入的权重长度必须和一维数组长度（即序列点数）相等')
    if abs(sum(w)-1) > 0.001:
        raise Exception('weights are not useable')
    return np.matmul(df.values, w), w


k_mul = np.array(range(7))
y_all = [[]] * len(k_mul)

for i in range(0, len(k_mul)):
    y_all[i] = np.sin(random.choices(range(0, 100), k=20))

y_all = pd.DataFrame(y_all).T
y_all.fillna(value=0, inplace=True)


print(dyn_df_weighted(y_all, type='geometric'))
print(dyn_seri_weighted(y_all.loc[20-1, :], type='geometric'))
print(abs(dyn_df_weighted(y_all, type='geometric')[0][-1] - dyn_seri_weighted(y_all.loc[20-1, :], type='geometric')[0]) < 1e-4, '\n')

print(dyn_df_weighted(y_all, type='arithmetic'))
print(dyn_seri_weighted(y_all.loc[20-1, :], type='arithmetic'))
print(abs(dyn_df_weighted(y_all, type='arithmetic')[0][-1] - dyn_seri_weighted(y_all.loc[20-1, :], type='arithmetic')[0]) < 1e-4, '\n')

print(dyn_df_weighted(y_all, w=np.ones(len(k_mul)), type='arithmetic'))
print(dyn_seri_weighted(y_all.loc[20-1, :], w=np.ones(len(k_mul)), type='arithmetic'))
print(abs(dyn_df_weighted(y_all, w=np.ones(len(k_mul)), type='arithmetic')[0][-1] - dyn_seri_weighted(y_all.loc[20-1, :], w=np.ones(len(k_mul)), type='arithmetic')[0]) < 1e-4, '\n')

print(dyn_df_weighted(y_all, w=np.ones(len(k_mul))))
print(dyn_seri_weighted(y_all.loc[20-1, :], w=np.ones(len(k_mul))))
print(abs(dyn_df_weighted(y_all, w=np.ones(len(k_mul)))[0][-1] - dyn_seri_weighted(y_all.loc[20-1, :], w=np.ones(len(k_mul)))[0]) < 1e-4, '\n')
