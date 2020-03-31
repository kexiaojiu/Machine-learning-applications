# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:26:54 2020

@author: kejie
功能：
    k-means聚类
"""

# 导入库函数
import numpy as np
from sklearn.cluster import KMeans


# 加载数据，创建K-means算法实例，并进行训练，获得标签
if __name__ == '__main__':
    data,city_name = loadData('')