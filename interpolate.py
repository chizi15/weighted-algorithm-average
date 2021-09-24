import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import statsmodels.api as sm
import matplotlib.pyplot as plt


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


# 用于构造函数的坐标点
data_x = (1, 10, 20, 30)
data_y = (1e-5, 0.1, 0.9, 1)

for i in range(data_x[0], data_x[-1]):
    if s_curve_interp(i+1, data_x, data_y) - s_curve_interp(i, data_x, data_y) < 0:
        raise Exception('构造出的函数应不严格地单调递增，但此时在第 %s 个点处，s型曲线的值降低' % (i+1))
print('各个y坐标值：')
for i in range(data_x[0], data_x[-1]+1):
    print(s_curve_interp(i, data_x, data_y))

# x坐标间距越小，构造出的曲线就会显示得越光滑；因为配置的插值函数在临界点处左右极限和左右一阶导数相等，所以函数在整个定义域上连续且光滑
# Array of evenly spaced values. For floating point arguments, the length of the result is `ceil((stop - start)/step)`.
# Because of floating point overflow, this rule may result in the last element of `out` being greater than `stop`.
xnew = np.arange(data_x[0], data_x[-1], 0.01)
ynew = [s_curve_interp(i, x=data_x, y=data_y) for i in xnew]
plt.figure()
plt.plot(data_x, data_y, 'o', xnew, ynew, '-')
plt.title('constructed interpolate points')
plt.show()
# 根据构造的函数生成归一化的权重w。因为每个w的分子与构造曲线的每个y值完全相同，而每个w的分母都是sum(ynew)，
# 所以w的分布完全由其分子确定，而其分子的分布与构造曲线y值的分布相同，所以w的分布特征与构造曲线的分布特征完全相同。
plt.figure()
w = [i/sum(ynew) for i in ynew]
plt.plot(list(range(len(w))), w)
plt.title('weights')
plt.show()

#########################################################################################################
# 用scipy.interpolate插值
x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interp1d(x, y, bounds_error=True)
f2 = interp1d(x, y, kind='quadratic')

xnew = np.linspace(0, 10, num=41, endpoint=True)
plt.figure()
plt.plot(x, y, 'o', xnew, f(xnew), '--', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'quadratic'], loc='best')
plt.show()


# 用pd.Series.interpolate插值
dta = sm.datasets.co2.load_pandas().data.co2
plt.figure()
co2 = dta.interpolate(inplace=False)  # deal with missing values. see issue
co2.plot(color='r', label='interpolated')
dta.plot(color='g', label='origin')
plt.legend()
