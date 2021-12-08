from scipy.interpolate import CubicSpline
from sklearn import metrics
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn import metrics
from statsmodels.tools import eval_measures
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
plt.rc('font', size=10)
pd.set_option('display.max_columns', None)


# ###########---------------set up and plot input data-----------------######################
base_value = 10  # 设置level、trend、season项的基数
steps_day, steps_week = 1, 1
length = [steps_day*20+steps_day, steps_week*20+steps_week, steps_week*20+steps_week]*2  # 代表周、日序列对的长度

weights = []
for i in range(-base_value + 1, 1):
    weights.append(0.5 ** i)  # 设置y_level项随机序列的权重呈递减指数分布，底数越小，y_level中较小值所占比例越大。
weights = np.array(weights)


# #########################################################--构造乘法周期性时间序列，模拟真实销售；外层是list，内层的每一条序列是series
y_level_actual, y_trend_actual, y_season_actual, y_noise_actual, y_input_mul_actual = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_actual[i] = np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性
    y_level_actual[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(abs(y_season_actual[i])) + np.average(abs(y_season_actual[i]))  # 用指数权重分布随机数模拟水平项
    y_trend_actual[i] = (2 * max(y_season_actual[i]) + np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
        + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
        / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(y_level_actual[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_actual[i] = 3*np.random.standard_t(length[i]-1, length[i])  # 假定数据处于理想状态，并使噪音以乘法方式进入模型，则可令噪音在0附近呈学生分布。
    y_noise_actual[i][abs(y_noise_actual[i]) < max(y_noise_actual[i])*0.9] = 1  # 保留随机数中的离群值，将非离群值置为1
    y_input_mul_actual[i] = (y_level_actual[i] + y_trend_actual[i]) * y_season_actual[i] * y_noise_actual[i]  # 假定周期项以乘法方式组成输入数据

    print(f'第{i}条真实序列中水平项的极差：{max(y_level_actual[i]) - min(y_level_actual[i])}，均值：{np.mean(y_level_actual[i])}')
    print(f'第{i}条真实序列中趋势项的极差：{max(y_trend_actual[i]) - min(y_trend_actual[i])}，均值：{np.mean(y_trend_actual[i])}')
    print(f'第{i}条真实序列中周期项的极差：{max(y_season_actual[i]) - min(y_season_actual[i])}，均值：{np.mean(y_season_actual[i])}')
    print(f'第{i}条真实序列中噪音项的极差：{max(y_noise_actual[i]) - min(y_noise_actual[i])}，均值：{np.mean(y_noise_actual[i])}')
    print(f'第{i}条真实乘法性序列最终极差：{max(y_input_mul_actual[i]) - min(y_input_mul_actual[i])}，均值：{np.mean(y_input_mul_actual[i])}', '\n')

    y_level_actual[i] = pd.Series(y_level_actual[i]).rename('y_level_actual')
    y_trend_actual[i] = pd.Series(y_trend_actual[i]).rename('y_trend_actual')
    y_season_actual[i] = pd.Series(y_season_actual[i]).rename('y_season_actual')
    y_noise_actual[i] = pd.Series(y_noise_actual[i]).rename('y_noise_actual')
    y_input_mul_actual[i] = pd.Series(y_input_mul_actual[i]).rename('y_input_mul_actual')
    # y_input_mul_actual[i][y_input_mul_actual[i] < 0.011] = 0.011  # 将series中小于0.011的数置为0.011；因为后续regression_accuracy，regression_evaluation会将series中小于0的置为0.01，若此处不将小于0.011的置为0.011，则画出的图可能与后续两个综合评估函数中所使用的序列不一致。
    print('第{0}条真实序列的初始生成值：'.format(i))
    print(y_input_mul_actual[i], '\n')

##########################################################--构造乘法周期性时间序列，模拟预测销售；外层是list，内层的每一条序列是series
y_level_pred, y_trend_pred, y_season_pred, y_noise_pred, y_input_mul_pred = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_pred[i] = 1/2 * np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性，使预测销售的波动振幅比真实销售小
    y_level_pred[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(abs(y_season_pred[i])) + np.average(abs(y_season_pred[i]))  # 用指数权重分布随机数模拟水平项，使其相对于真实销售有所偏移
    y_trend_pred[i] = (2 * max(y_season_pred[i]) + np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
        + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
        / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(y_level_pred[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_pred[i] = np.random.standard_t(length[i]-1, length[i])  # 假定数据处于理想状态，并使噪音以乘法方式进入模型，则可令噪音在0附近呈学生分布；使其比真实销售的噪音小。
    y_noise_pred[i][abs(y_noise_pred[i]) < max(y_noise_pred[i])*0.9] = 1  # 保留随机数中的离群值，将非离群值置为1
    y_input_mul_pred[i] = (y_level_pred[i] + y_trend_pred[i]) * y_season_pred[i] * y_noise_pred[i]  # 假定周期项以乘法方式组成输入数据

    print(f'第{i}条预测序列中水平项的极差：{max(y_level_pred[i]) - min(y_level_pred[i])}，均值：{np.mean(y_level_pred[i])}')
    print(f'第{i}条预测序列中趋势项的极差：{max(y_trend_pred[i]) - min(y_trend_pred[i])}，均值：{np.mean(y_trend_pred[i])}')
    print(f'第{i}条预测序列中周期项的极差：{max(y_season_pred[i]) - min(y_season_pred[i])}，均值：{np.mean(y_season_pred[i])}')
    print(f'第{i}条预测序列中噪音项的极差：{max(y_noise_pred[i]) - min(y_noise_pred[i])}，均值：{np.mean(y_noise_pred[i])}')
    print(f'第{i}条预测乘法性序列最终极差：{max(y_input_mul_pred[i]) - min(y_input_mul_pred[i])}，均值：{np.mean(y_input_mul_pred[i])}', '\n')

    y_level_pred[i] = pd.Series(y_level_pred[i]).rename('y_level_pred')
    y_trend_pred[i] = pd.Series(y_trend_pred[i]).rename('y_trend_pred')
    y_season_pred[i] = pd.Series(y_season_pred[i]).rename('y_season_pred')
    y_noise_pred[i] = pd.Series(y_noise_pred[i]).rename('y_noise_pred')
    y_input_mul_pred[i] = pd.Series(y_input_mul_pred[i]).rename('y_input_mul_pred')
    # y_input_mul_pred[i][y_input_mul_pred[i] < 0.011] = 0.011  # 将series中小于0.011的数置为0.011；因为后续regression_accuracy，regression_evaluation会将series中小于0的置为0.01，若此处不将小于0.011的置为0.011，则画出的图可能与后续两个综合评估函数中所使用的序列不一致。
    print('第{0}条预测序列的初始生成值：'.format(i))
    print(y_input_mul_pred[i], '\n')

# # 绘制真实值和对应的预测值序列
# for i in range(len(y_input_mul_actual)):
#     plt.figure('origin true and predict series {0}'.format(i), figsize=(5,10))
#     ax1 = plt.subplot(5,1,1)
#     ax2 = plt.subplot(5,1,2)
#     ax3 = plt.subplot(5,1,3)
#     ax4 = plt.subplot(5,1,4)
#     ax5 = plt.subplot(5,1,5)
#     y_input_mul_actual[i].plot(ax=ax1, legend=True)
#     y_level_actual[i].plot(ax=ax2, legend=True)
#     y_trend_actual[i].plot(ax=ax3, legend=True)
#     y_season_actual[i].plot(ax=ax4, legend=True)
#     y_noise_actual[i].plot(ax=ax5, legend=True)
#     y_input_mul_pred[i].plot(ax=ax1, legend=True)
#     y_level_pred[i].plot(ax=ax2, legend=True)
#     y_trend_pred[i].plot(ax=ax3, legend=True)
#     y_season_pred[i].plot(ax=ax4, legend=True)
#     y_noise_pred[i].plot(ax=ax5, legend=True)

rng = np.random.default_rng()
# mu, sigma = 30., 3.  # mean and standard deviation
# x = rng.lognormal(mu, sigma, size=500)
#
# # for i in range(len(y_input_mul_actual)):
# for i in [0]:
#     # y = y_input_mul_actual[i].values
#     y = x
#     yn = geo_zscore(y)
#     yn1, lamda = stats.yeojohnson(yn)
#
#     fig, ax = plt.subplots(1, 1)
#     ax.hist(y, density=True, histtype='stepfilled', alpha=0.2)
#     fig1, ax1 = plt.subplots(1, 1)
#     ax1.hist(yn, density=True, histtype='stepfilled', alpha=0.2)
#     fig1, ax1 = plt.subplots(1, 1)
#     ax1.hist(yn1, density=True, histtype='stepfilled', alpha=0.2)
#
#     skewness1 = stats.skew(y)
#     skewness2 = stats.skew(yn)
#     skewness3 = stats.skew(yn1)
#     print('skewy:', skewness1)
#     print('skewyn:', skewness2)
#     print('skewyn1:', skewness3)
#
#     k2, p = stats.normaltest(y)
#     alpha = 1e-3
#     print("p = {:g}".format(p))
#     if p < alpha:  # null hypothesis: x comes from a normal distribution
#         print("The null hypothesis can be rejected")
#     else:
#         print("The null hypothesis cannot be rejected")
#
#     k2, p = stats.normaltest(yn)
#     alpha = 1e-3
#     print("p = {:g}".format(p))
#     if p < alpha:  # null hypothesis: x comes from a normal distribution
#         print("The null hypothesis can be rejected")
#     else:
#         print("The null hypothesis cannot be rejected")
#
#     k2, p = stats.normaltest(yn1)
#     alpha = 1e-3
#     print("p = {:g}".format(p))
#     if p < alpha:  # null hypothesis: x comes from a normal distribution
#         print("The null hypothesis can be rejected", '\n')
#     else:
#         print("The null hypothesis cannot be rejected", '\n')


def geo_zscore(samples, bias=1, ddof=1):
    """
    gzscore = log(a/gmu) / log(gsigma), where gmu (resp. gsigma) is the geometric mean (resp. standard deviation).
    """
    # The geometric standard deviation is defined for strictly positive values only, because of log.
    samples += abs(min(samples)) + bias
    # gstd = exp(std(log(a)))
    geo_std = np.exp(np.std(np.log(samples), ddof=ddof))
    # let degrees of freedom correction in the calculation of the standard deviation to be 0.
    gzscore = np.log(samples / stats.gmean(samples)) / np.log(geo_std)

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
    传入一维数组seri，可以是series,array,list,tuple；若type='geometric'或'arithmetic'，且输入了w，则w不起作用；若不输入权重，则根据seri的长度动态计算基于几何级数或算数级数再作归一化的权重，再做算术平均；
    也可人为输入权重做算术平均；若不输入type和w，则进行简单算数平均；因为使用np.dot，则seri索引越小，权重越大；将seri各点与权重相乘再相加，得到一个最终点。
    :param seri: 需要进行加权变成一个点的一维数组
    :param type: 采用几何级数或算数级数进行加权，或人为指定权重，或默认权重相等，type = 'geometric'或'arithmetic'或None；若type='geometric'或'arithmetic'，且输入了w，则w不起作用。
    :param w: 一维的权重系数，可以是series,array,list,tuple；若手动输入，其长度必须和一维数组seri（即序列点数）相等
    :param r: 指定几何级数分母的公比
    :param d: 指定算数级数分母的公差
    :param initial: 指定算数级数分母的初始值
    :return: seri各点与权重w相乘再相加，返回的一个加权后的最终点
    """
    if type not in ['amean_geo', 'amean_arith', 'amean_trim', 'amean_sigmoid', 'gmean', 'hmean', 'smean', 'normean', None]:
        raise Exception('type must be one of the \'amean_geo\', \'amean_arith\', \'amean_trim\', \'amean_sigmoid\', \'gmean\', \'hmean\', \'smean\', \'normean\', or \'None\'')
    elif w is None:
        w = np.ones(len(seri)) / sum(np.ones(len(seri)))  # 生成均等权重
    if len(w) != len(seri):
        raise Exception('len(w) != len(seri)')
    elif type in ['amean_geo', 'amean_arith']:  # weighted arithmetic average, weights are geometric series or arithmetic series
        w = list()
        if type == 'amean_geo':
            for i in range(len(seri)):
                w.append(initial * (1 / r) ** i)  # 生成首项是initial，公比是(1/r)的几何级数作权重；权重从左至右呈指数型降低
        else:
            for i in range(len(seri)):
                w.append(1 / (initial + d * i))  # 生成首项是initial，公差是d的算术级数，再做倒数作为权重；权重从左至右呈指数型降低
        w = np.array(w) / sum(w)
        if abs(sum(w) - 1) > 0.001:
            raise Exception('weights are not useable')
        return np.dot(np.array(seri), w)
    elif type is 'amean_trim':
        if low < 0 or low > len(seri) - 1 or up < 1 or up > len(seri) or (not isinstance(low, int)) or (
        not isinstance(up, int)):
            raise Exception('low is index from the start, up is index from the end, and must be \'int\'')
        seri.sort()
        return stats.tmean(seri, (seri[low], seri[-up]), inclusive=(True, True))
    elif type is 'amean_sigmoid':  # 权重从左至右呈S型升高
        # 用于构造函数的坐标点
        critical_x = (1, len(seri) / 3, len(seri) * 2 / 3, len(seri))
        critical_y = critical_y
        xnew = np.arange(critical_x[0], critical_x[-1] + 1)
        ynew = [s_curve_interp(i, x=critical_x, y=critical_y) for i in xnew]
        # 根据构造的函数生成归一化的权重w。因为每个w的分子与构造曲线的每个y值完全相同，而每个w的分母都是sum(ynew)，
        # 所以w的分布完全由其分子确定，而其分子的分布与构造曲线y值的分布相同，所以w的分布特征与构造曲线的分布特征完全相同。
        w = [i / sum(ynew) for i in ynew]
        return np.dot(np.array(seri), w)
    elif type is 'gmean':
        return stats.gmean(seri, weights=w)  # simple geometric average, or weighted geometric average
    elif type is 'hmean':
        return stats.hmean(seri)  # simple harmonic average
    elif type is 'smean':
        return metrics.mean_squared_error(seri, np.zeros(len(seri)), sample_weight=w, squared=False)  # RMSE with 0
    elif type is 'normean':
        samp_geo = geo_zscore(seri)
        samp_yj = stats.yeojohnson(seri)
        samp_geo_yj = stats.yeojohnson(samp_geo)
        nt_samp_yj = stats.normaltest(samp_yj)
        nt_samp_geo_yj = stats.normaltest(samp_geo_yj)

        if nt_samp_yj[1] > nt_samp_geo_yj[1]:
            skew_yj = stats.skew(samp_yj)
            weights = stats.skewnorm.pdf(samp_yj, skew_yj)[1]
            w = np.array(weights) / sum(weights)
            print('nt_samp_yj', w)
        else:
            pass
            # w = np.array(nt_samp_geo_yj) / sum(nt_samp_geo_yj)
            # print('nt_samp_geo_yj', w)
        return np.dot(np.array(seri), w)
    elif type is None:
        w = np.array(w) / sum(w)  # 自定义权重
        return np.dot(np.array(seri), w)


samp_geo = geo_zscore(y_input_mul_actual[0])
samp_yj, lda1 = stats.yeojohnson(y_input_mul_actual[0])

plt.figure()
ax1 = plt.subplot(3,1,1)
ax2 = plt.subplot(3,1,2)
ax3 = plt.subplot(3,1,3)
y_input_mul_actual[0].plot(ax=ax1, legend=True)
samp_geo.plot(ax=ax2)
pd.Series(samp_yj).plot(ax=ax3)

plt.figure()
ax1 = plt.subplot(3,1,1)
ax2 = plt.subplot(3,1,2)
ax3 = plt.subplot(3,1,3)
y_input_mul_actual[0].hist(ax=ax1, density=False)
samp_geo.hist(ax=ax2, density=True)
pd.Series(samp_yj).hist(ax=ax3, density=True)

nt_samp_yj = stats.normaltest(samp_yj)
nt_samp_geo_yj = stats.normaltest(samp_geo_yj)

plt.figure()
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)

if nt_samp_yj[1] > nt_samp_geo_yj[1]:
    skew_yj = stats.skew(samp_yj)
    weights = stats.skewnorm.pdf(samp_yj, skew_yj, loc=np.mean(samp_yj), scale=np.std(samp_yj, ddof=0))
    pd.Series(weights).plot(ax=ax1)
    pd.Series(weights).hist(ax=ax2)

    w = np.array(weights) / sum(weights)
    print('nt_samp_yj', w)
else:
    w = np.array(nt_samp_geo_yj) / sum(nt_samp_geo_yj)
    print('nt_samp_geo_yj', w)

np.dot(np.array(y_input_mul_actual[0]), w)


#
# samp_yj, lda1 = stats.yeojohnson(y_input_mul_actual[0])
# samp_geo_yj, lda2 = stats.yeojohnson(samp_geo)
# nt_samp_yj = stats.normaltest(samp_yj)
# nt_samp_geo_yj = stats.normaltest(samp_geo_yj)
#
# if nt_samp_yj[1] > nt_samp_geo_yj[1]:
#     skew_yj = stats.skew(samp_yj)
#     weights = stats.skewnorm.pdf(samp_yj, skew_yj)
#     w = np.array(weights) / sum(weights)
#     print('nt_samp_yj', w)
# else:
#     w = np.array(nt_samp_geo_yj) / sum(nt_samp_geo_yj)
#     print('nt_samp_geo_yj', w)
#
# np.dot(np.array(y_input_mul_actual[0]), w)
'amean_geo', 'amean_arith', 'amean_trim', 'amean_sigmoid', 'gmean', 'hmean', 'smean', 'normean', None
x = np.arange(1, 11)
dyn_seri_weighted(x, type='smean')
