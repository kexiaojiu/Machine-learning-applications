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

# 定义函数 
## 读取数据
def load_data(file_name):
    fr = open(file_name,'r+')
    ## 一次读取整个文件
    lines = fr.readlines()
    ## 存放城市各类消费信息
    ret_data = []
    ## 城市名称
    ret_city_name = []
    
    for line in lines:
        items = line.strip().split(",")
        ret_city_name.append(items[0])
        ret_data.append([ float(items[i]) for i in range(1,len(items))])
            
    return ret_data,ret_city_name


# 加载数据，创建K-means算法实例，并进行训练，获得标签
if __name__ == '__main__':
    # 定义参数
    ## 文件参数
    file_path_input = '..\\02 数据\\聚类\\'
    file_name_input = '31省市居民家庭消费水平-city.txt'
    ## 聚类参数
    #n_cluster_value = 3
    ## 存放聚类数据
    #city_cluster = [[],[],[]]
    
    ## 输入分类
    n_cluster_value = int(input("请确定需要分几类？："))
    city_cluster = []
    for i in range(n_cluster_value):
        city_cluster.append([])
    
    # 载入数据
    data,city_name = load_data(file_path_input+file_name_input)
 
    # 创建kmeans实例
    km_example = KMeans(n_clusters=n_cluster_value)
    
    # 打标签
    label = km_example.fit_predict(data)
    expenses = np.sum(km_example.cluster_centers_,axis=1)
    #print(expenses)
    
    for i in range(len(city_name)):
        #print(city_name[i])
        city_cluster[label[i]].append(city_name[i])
        
    for i in range(len(city_cluster)):
        print("Expenses:%.2f"%expenses[i])
        print(city_cluster[i])

