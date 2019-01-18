"""
主要用来验证正态分布函数是否正确
"""
import numpy as np
from scipy import stats


def gaussian(_mean, _std, x):
    coefficient = 1 / (_std * np.sqrt(2 * np.pi))
    square_std = 2 * _std * _std
    print(x, _mean, _std, coefficient * np.exp(-np.square(x - _mean) / square_std))


if __name__ == '__main__':
    gaussian(4.9615, 0.3512, 7.2)
    print(stats.norm.pdf(7.2, 4.9615, 0.3512))
