import numpy as np
import pandas as pd
import random


def dyn_df_weighted(df, type, w=None, r=3/2, d=1, initial=1):
    """
    传入df，根据df的列数动态计算基于几何级数或算数级数再作归一化的权重，因为使用np.matmul，则在df的列中，索引越小的权重越大；将df各列与权重相乘再相加，得到一条最终序列。
    :param df: 需要加权的各列组成的df
    :param type: 采用几何级数或算数级数加权，type = 'geometric'或'arithmetic'或None
    :param w: 权重系数可人为指定，默认为None
    :param r: 指定几何级数分母的公比
    :param d: 指定算数级数分母的公差
    :param initial: 指定算数级数分母的初始值
    :return: 将df各列分别乘以权重w再相加得到一条最终的加权序列
    """
    if w is None:
        w = list()
        if type == 'geometric':
            for i in range(len(df.columns)):
                w.append((1 / r) ** i)
        elif type == 'arithmetic':
            for i in range(len(df.columns)):
                w.append(1 / (initial + d * i))
        else:
            raise Exception('if w=None, type must be one of geometric or arithmetic')
        w = np.array(w) / sum(w)
    elif (w is not None) and (type is None) and (len(w) == len(df.columns)):
        w = np.array(w) / sum(w)
    else:
        raise Exception('手动输入的权重长度必须和df列数（即序列条数）相等，或者必须选择type')
    print(w, sum(w))
    return np.matmul(df.values, w)


def dyn_seri_weighted(seri, type, w=None, r=2, d=1, initial=1):
    """
    传入seri，根据seri的长度动态计算基于几何级数或算数级数再作归一化的权重，因为使用np.dot，则seri索引越小，权重越大；将seri各点与权重相乘再相加，得到一条最终seri。
    :param seri: 需要加权的一维数组
    :param type: 采用几何级数或算数级数加权，type = 'geometric'或'arithmetic'或None
    :param w: 权重系数可人为指定，默认为None
    :param r: 指定几何级数分母的公比
    :param d: 指定算数级数分母的公差
    :param initial: 指定算数级数分母的初始值
    :return: 返回seri各点与权重w相乘再相加，得到的一条最终加权序列
    """
    if w is None:
        w = list()
        if type == 'geometric':
            for i in range(len(seri)):
                w.append((1 / r) ** i)
        elif type == 'arithmetic':
            for i in range(len(seri)):
                w.append(1 / (initial + d * i))
        else:
            raise Exception('if w=None, type must be one of geometric or arithmetic')
        w = np.array(w) / sum(w)
    elif (w is not None) and (type is None) and (len(w) == len(seri)):
        w = np.array(w) / sum(w)
    else:
        raise Exception('手动输入的权重长度必须和一维数组长度（即序列点数）相等，或者必须选择type')
    print(w, sum(w))
    return np.dot(seri, w)


k_mul = np.array(range(7))
y_all = [[]] * len(k_mul)

for i in range(0, len(k_mul)):
    y_all[i] = np.sin(random.choices(range(0, 100), k=20))

y_all = pd.DataFrame(y_all).T
y_all.fillna(value=0, inplace=True)

print(dyn_df_weighted(y_all, type='geometric'), '\n')
print(dyn_seri_weighted(y_all.loc[20-1, :], type='geometric'), '\n')

print(dyn_df_weighted(y_all, type='arithmetic'), '\n')
print(dyn_seri_weighted(y_all.loc[20-1, :], type='arithmetic'), '\n')

print(dyn_df_weighted(y_all, w=np.ones(len(k_mul)), type='arithmetic'), '\n')
print(dyn_seri_weighted(y_all.loc[20-1, :], w=np.ones(len(k_mul)), type='geometric'), '\n')

data = pd.read_excel('/Users/zc/Desktop/test.xlsx')
dyn_df_weighted(data, type='arithmetic')
