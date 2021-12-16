import numpy as np
from sklearn import metrics
from scipy import stats
from scipy.interpolate import CubicSpline


def geo_zscore(samples, bias=1, ddof=1):
    """
    gzscore = log(a/gmu) / log(gsigma), where gmu (resp. gsigma) is the geometric mean (resp. standard deviation).
    """
    # The geometric standard deviation is defined for strictly positive values only, because of log.
    # 特别重要：不要写成samples += abs(min(samples)) + bias，否则当同一个序列第二次调用geo_zscore时，这条序列已经被改变，不是原序列了，
    # 体现为函数对输入变量的"全局"赋值作用；或者说每调用一次函数，输入序列samples就被"全局"改变一次。
    samp_shift = samples + abs(min(samples)) + bias  # 使samp_shift>0，满足下面np.log的要求
    # gstd = exp(std(log(a)))
    geo_std = np.exp(np.std(np.log(samp_shift), ddof=ddof))
    # let degrees of freedom correction in the calculation of the standard deviation to be 0.
    gzscore = np.log(samp_shift / stats.gmean(samp_shift)) / np.log(geo_std)

    return gzscore


def s_curve_interp(n, x=(1, 10, 20, 30), y=(1e-5, 0.1, 0.9, 1)):
    """
    n：需要根据构造的插值函数得到对应y值的x坐标
    x：用于构造插值函数的点的x坐标，n值最好在x的范围内，因为插值函数不合适做外推
    y：用于构造插值函数的点的y坐标；x和y是成对的坐标，遵循奥卡姆剃刀原则，最少只需四个点，即三段插值函数，就可以构造任意大致规律的全局函数；若点数越多，构造出的函数形态就可以控制得越细致。
    return: 构造出的插值函数的x坐标为n时，对应的一个y坐标值
    """
    if x[0] <= n < x[1]:
        cs1 = CubicSpline(x[:2], y[:2], bc_type=((1, y[1] / x[1]**2), (1, y[1] / x[1]**0.5)), extrapolate=False)
        r = cs1(n)
        if r < 0:
            r = cs1(x[0]+1)
    elif x[1] <= n < x[2]:
        cs2 = CubicSpline(x[1:3], y[1:3], bc_type=((1, y[1] / x[1]**0.5), (1, (y[2]-y[1]) / (x[2]-x[1])**2)), extrapolate=False)
        r = cs2(n)
    else:
        cs3 = CubicSpline(x[-2:], y[-2:], bc_type=((1, (y[2] - y[1]) / (x[2] - x[1]) ** 2), (1, (y[3] - y[2]) / (x[3] - x[2]) ** 2)), extrapolate=False)
        r = cs3(n)
        if r > 1:
            r = cs3(x[-1]-1)
    return float(r)


def dyn_seri_weighted(seri, type=None, w=None, initial=1, r=2, d=1, low=0, up=1, critical_y=(1e-5, 0.1, 0.9, 1)):
    """
    :param seri: 需要进行加权平均变成一个值的一维数组，可以是series,array,list,tuple；在调用不同type时需注意可能有不同的顺序。
    :param type: 选择序列seri加权的类型，'amean_geo', 'amean_arith', 'amean_trim', 'amean_sigmoid', 'gmean', 'hmean',
    'smean', 'normean', None。
     其中，
    'amean_geo'是权重从左至右呈几何级数递减的算术平均；
    'amean_arith'是权重从左至右呈算术级数递减的算术平均；
    'amean_trim'是两侧截尾简单算术平均，是按小于临界值和大于临界值去截断，而不是按顺序的索引截断；
    'amean_sigmoid'是权重从左至右呈S型升高的算术平均，即加权结果不太受序列左侧点的影响，而受右侧点影响比较大；
    'normean'是基于偏斜正态概率的加权算术平均，受两侧离群值影响小，因为假定数据出现离群值的概率小；
     None是对序列做简单算术平均或权重为w的加权算术平均。
    'gmean'是简单几何平均或者权重为w的加权几何平均，比算数平均更接近较小值；
    'hmean'是简单调和平均，结果比'gmean'更趋近较小值；
    'smean'是均方根，比算术平均更接近较大值；
    :param w: 一维的权重系数，可以是series,array,list,tuple；若手动输入，其长度必须和一维数组seri（即序列点数）相等
    :param r: type='amean_geo'时，指定几何级数分母的公比
    :param d: type='amean_arith'时，指定算数级数分母的公差
    :param initial: type='amean_arith'时，指定算数级数分母的初始值
    :param low: type='amean_trim'，截尾简单算数平均中，从小到大，截断比第low个值更小的那些值，但不包括第low个较小值
    :param up: type='amean_trim'，截尾简单算数平均中，从大到小，截断比第up个值更大的那些值，但不包括第up个较大值
    :param critical_y: type='amean_sigmoid'中，设置S型曲线在四个临界点处的y值，相邻两个y值间的差值δy越大，则在该区间内曲线上升越快；δy越小，上升越慢。
    :return: seri各点与权重w相乘再相加，返回的一个加权后的最终值
    """
    seri = np.array(seri)
    if type not in ['amean_geo', 'amean_arith', 'amean_trim', 'amean_sigmoid', 'gmean', 'hmean', 'smean', 'normean', None]:
        raise Exception('type must be one of the \'amean_geo\', \'amean_arith\', \'amean_trim\', \'amean_sigmoid\', '
                        '\'gmean\', \'hmean\', \'smean\', \'normean\', or \'None\'')
    elif w is None:
        w = np.ones(len(seri)) / sum(np.ones(len(seri)))  # 生成均等权重
    if len(w) != len(seri):
        raise Exception('len(w) != len(seri)')
    # weighted arithmetic average, weights are geometric series or arithmetic series
    elif type in ['amean_geo', 'amean_arith']:
        w = list()
        if type == 'amean_geo':
            for i in range(len(seri)):
                w.append(initial * (1 / r) ** i)  # 生成首项是initial，公比是(1/r)的几何级数作权重；权重从左至右呈指数型降低
        else:
            for i in range(len(seri)):
                w.append(1 / (initial + d * i))  # 生成首项是initial，公差是d的算术级数，再做倒数作为权重；权重从左至右呈指数型降低
        w = np.array(w) / sum(w)
        return np.dot(np.array(seri), w)
    elif type == 'amean_trim':
        if low < 0 or low > len(seri) - 1 or up < 1 or up > len(seri) or (not isinstance(low, int)) \
                or (not isinstance(up, int)):
            raise Exception('low is index from the start, up is index from the end, and must be \'int\'')
        seri.sort()
        return stats.tmean(seri, (seri[low], seri[-up]), inclusive=(True, True))
    elif type == 'amean_sigmoid':  # 权重从左至右呈S型升高，即加权结果不太受左侧点的影响，而受右侧点影响比较大
        # 用于构造函数的坐标点
        critical_x = (1, len(seri) / 3, len(seri) * 2 / 3, len(seri))  # 将x轴上的定义域分为均等的三段
        critical_y = critical_y
        xnew = np.arange(critical_x[0], critical_x[-1] + 1, 1)  # 设置每个需要计算权重的x坐标
        ynew = [s_curve_interp(i, x=critical_x, y=critical_y) for i in xnew]  # 计算这些x坐标在S曲线上对应的y值
        # 根据构造的函数生成归一化的权重w。因为每个w的分子与构造曲线的每个y值完全相同，而每个w的分母都是sum(ynew)，
        # 所以w的分布完全由其分子确定，而其分子的分布与构造曲线y值的分布相同，所以w的分布特征与构造曲线的分布特征完全相同。
        w = [i / sum(ynew) for i in ynew]
        return np.dot(np.array(seri), w)
    elif type == 'gmean':
        return stats.gmean(seri, weights=w)  # simple geometric average, or weighted geometric average
    elif type == 'hmean':
        return stats.hmean(seri)  # simple harmonic average
    elif type == 'smean':
        return metrics.mean_squared_error(seri, np.zeros(len(seri)), sample_weight=w, squared=False)  # RMSE with 0
    elif type == 'normean':
        samp_geo = geo_zscore(seri)  # 对序列seri做几何标准化变换，将接近幂律分布的序列转化为接近正态分布
        samp_yj, lmda_yj = stats.yeojohnson(seri)  # 对序列seri做幂变换，增强序列的正态性
        samp_geo_yj, lmda_geo_yj = stats.yeojohnson(samp_geo)  # 对序列samp_geo做幂变换，增强其正态性
        # 计算normalize后序列的正态性指标，k2和p-value，k2=skewness^2+kurtosis^2，越接近0正态性越强；p越接近1表明越有可能是从正态总体中抽样得到的序列samp_yj或samp_geo_yj
        nt_samp_yj = stats.normaltest(samp_yj)
        nt_samp_geo_yj = stats.normaltest(samp_geo_yj)
        if (nt_samp_yj[0] < nt_samp_geo_yj[0]) and (nt_samp_yj[1] > nt_samp_geo_yj[1]):  # 表示序列nt_samp_yj的正态性大于nt_samp_geo_yj
            skew_yj = stats.skew(samp_yj)  # 计算序列samp_yj的偏度
            # 最关键的一步，得到变换后的序列samp_yj的概率密度函数pdf，而samp_yj的顺序与传入序列seri的顺序完全相同，而与samp_yj的pdf函数图形上各自变量x的位置无关，即使自变量x就是samp_yj中的各个元素；
            # 则用samp_yj算出的pdf的y值，就可作为原序列seri对应各个点出现的概率，归一化后就可作为原序列seri的权重。
            pdf_y = stats.skewnorm.pdf(samp_yj, skew_yj, loc=np.mean(samp_yj), scale=np.std(samp_yj, ddof=0))
            w = np.array(pdf_y) / sum(pdf_y)
        else:
            skew_geo_yj = stats.skew(samp_geo_yj)
            pdf_y = stats.skewnorm.pdf(samp_geo_yj, skew_geo_yj, loc=np.mean(samp_geo_yj), scale=np.std(samp_geo_yj, ddof=0))
            w = np.array(pdf_y) / sum(pdf_y)
        return np.dot(np.array(seri), w)
    elif type is None:  # 简单或加权算术平均
        w = np.array(w) / sum(w)  # 自定义权重
        return np.dot(np.array(seri), w)


def dyn_df_weighted(df, type=None, w=None, initial=1*2, r=2, d=1/2):
    """
    传入二维数组df；若type='geometric'或'arithmetic'，且输入了w，则w不起作用；若不输入权重，则根据df的列数动态计算基于几何级数或算数级数再作归一化的权重，再做算术平均；
    也可人为输入权重做算术平均；若不输入type和w，则进行简单算数平均；因为使用np.matmul，则df.columns的索引越小，权重越大；将df的各列与权重相乘再相加，得到一条最终的序列。
    :param df: 需要进行加权变成一条序列的二维数组，df的每列代表一条需要进行加权的序列
    :param type: 采用几何级数或算数级数进行加权，或人为指定权重，或默认权重相等，type = 'geometric'或'arithmetic'或None；若type='geometric'或'arithmetic'，且输入了w，则w不起作用。
    :param w: 一维的权重系数，可以是series,array,list,tuple；若手动输入，其长度必须和一维数组seri（即序列点数）相等
    :param r: 指定几何级数分母的公比
    :param d: 指定算数级数分母的公差
    :param initial: 指定算数级数分母的初始值
    :return: df各列与权重w相乘再相加，返回一条最终的序列
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
    return np.matmul(df.values, w)


x = [1]*20 + [2]*10 + [3]*5 + [100]*2
print('原始样本：', x)
print('简单算术平均：', dyn_seri_weighted(x, type='amean_trim'))
print('几何级数算术平均，权重左大右小：', dyn_seri_weighted(x, type='amean_geo'))
print('算术级数算术平均，权重左大右小：', dyn_seri_weighted(x, type='amean_arith'))
print('S型曲线算术平均，权重左小右大：', dyn_seri_weighted(x, type='amean_sigmoid'))
print('正态概率算术平均：', dyn_seri_weighted(x, type='normean'))
print('加权算术平均，权重左小右大:', dyn_seri_weighted(x, type=None, w=np.arange(len(x))))
print('简单调和平均：', dyn_seri_weighted(x, type='hmean'))
print('简单几何平均：', dyn_seri_weighted(x, type='gmean'))
print('加权几何平均，权重左小右大：', dyn_seri_weighted(x, type='gmean', w=np.arange(len(x))))
print('简单均方根：', dyn_seri_weighted(x, type='smean'))
