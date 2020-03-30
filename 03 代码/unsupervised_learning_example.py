# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:17:53 2020

@author: kejie
功能：
    无监督学习导入数据集示例
"""
# 导入函数库
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)

data,target = load_boston(return_X_y=True)
print(data.shape)
print(target.shape)

from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data.shape)

data,target = load_iris(return_X_y=True)
print(data.shape)
print(target.shape)
list(iris.target_names)

from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
print(digits.target.shape)
print(digits.images.shape)

import matplotlib.pyplot as plt
plt.matshow(digits.images[1])
plt.show()
