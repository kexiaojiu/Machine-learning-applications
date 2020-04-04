# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:43:37 2020

@author: kejie
功能:
    DBSAVN聚类
"""

# 导入库
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# 定义函数
def read_data(file_name):
    mac2id = dict()
    online_times = []
    ## 读取数据
    fr = open(file_name,'r+',encoding='utf-8')
    lines = fr.readlines()
    ## 构建数据结构
    for line in lines:
        mac = line.split(',')[2]
        online_time = int(line.split(',')[6])
        start_time = int(line.split(',')[4].split(' ')[1].split(':')[0])
        if mac not in mac2id:
            mac2id[mac] = len(online_times)
            online_times.append((start_time,online_time))
        else:
            online_times[mac2id[mac]] = [(start_time,online_time)]
        
    real_X = np.array(online_times).reshape((-1,2))  
    return real_X



# 主模块
if __name__ == '__main__':
    # 定义参数
    ## 文件参数
    file_path_input = '..\\02 数据\\聚类\\'
    file_name_input = '学生月上网时间分布-TestData.txt'
    
    data = read_data(file_path_input+file_name_input)
    
    # 调用DBSCAN算法
    ## 在线小时
    online_hour_num = data[:,0:1]
    ## 创建DBSCAN实例
    db = DBSCAN(eps=0.01,min_samples=20).fit(online_hour_num)
    ## 数据的簇标签
    labels = db.labels_
    
    ## 打印数据被标记的标签，计算标签为-1，即噪声数据的比例
    print('Labels: ')
    print(labels)
    noise_ratio = len(labels[labels[:]==-1]) / len(labels)
    print('Noise ratio:',format(noise_ratio,'.2%'))
    
    ## 计算簇的个数并打印，评价聚类效果
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters:%d ' %n_clusters)
    print('Silgouette Coefficient:%0.3f' %metrics.silhouette_score(online_hour_num,labels))
    
    ## 打印各簇标号和各簇数据
    for i in range(n_clusters):
        print('Cluster ',i,':')
        print(list(online_hour_num[labels == i].flatten()))
    
    plt.hist(online_hour_num,24)