[toc]

# 1 机器学习简介

## 1.1 机器学习分类

- 监督学习（Supervised Learning）
- 无监督学习（Unsupervised Learning）
- 强化学习(Reinforcement Learning)
- 半监督学习（Semi-supervised Learning）
- 深度学习(Deep Learning)

## 1.2 Python sk-learn

- 一组简单有效的工具集
- 依赖Python的Numpy、SciPy和matplotlib
- 开源

## 1.3 sk-learn常用函数

| 类型                    | 应用               | 算法          |
| ----------------------- | ------------------ | ------------- |
| 分类Classification      | 异常检测，图形识别 | KNN,SVN       |
| 聚类Clustering          | 图形分割，群体划分 | K-Mean,谱聚类 |
| 回归Regression          | 价格预测，趋势预测 | 线性回归，SVR |
| 降维Dimension Reduction | 可视化             | PCA，NMF      |

## 1.4 学习目标

- 了解基本的机器学习原理和算法
- 学习利用机器学习算法解决应用问题的能力
- 掌握sklearn库中常用的机器学习算法的基本调用方法

# 2 sk-learn库

安装依赖库Numpy,SciPy,matplotlib

## 2.1 常用数据集

|          | 数据集名称             | 调用方式               | 适用算法   | 数据规模     |
| -------- | ---------------------- | ---------------------- | ---------- | ------------ |
| 小数据集 | 波士顿房价数据集       | load_boston()          | 回归       | 506*13       |
|          | 鸢尾花数据集           | load_iris()            | 分类       | 150*4        |
|          | 糖尿病数据集           | load_diabetes()        | 回归       | 442*10       |
|          | 手写字体数据集         | load_digits()          | 分类       | 5620*64      |
|          | Olivetti脸部图像数据集 | fetch_olivetti_faces() | 降维       | 400*64*64    |
|          | 新闻分类数据集         | fetch_20newsgroups()   | 分类       | ---          |
|          | 带标签的人脸数据集     | fetch_lfw_people()     | 分类；降维 | ---          |
|          | 路透社新闻语料数据集   | fetch_rcvl()           | 分类       | 804414*47236 |

* 注：小数据集可以直接使用，大数据集需要在调用时候程序自动下载（一次即可）

## 2.2 波士顿房价数据集

### 2.2.1 简介

包含506组数据，，每条包含房屋以及房屋周围的详细信息，其中包括城镇犯罪率、一氧化碳浓度、住宅平均房间数、到中心区域的加权距离以及自住房平均房价等。

### 2.2.2 使用方法

使用sklearn.datasets.load_boston

重要参数：

* return_X_y:表示是否返回target(价格)，默认为False，只返回data(即属性)

```python
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)

data,target = load_boston(return_X_y=True)
print(data.shape)
print(target.shape)
```

输出依次为：

```
>(506,13)
>(506,13)
>(503,)
```



## 2.3 鸢尾花数据集

### 2.3.1 简介

测量数据以及所属的类别。

测量数据包括：萼片长度、萼片宽度、花瓣长度、花瓣宽度

类别包括Iris Setosa,Iris Versicolour,Iris Virginica

### 2.3.2 使用方法

使用sklearn.datasets.load_iris

重要参数：

* return_X_y:表示是否返回target(价格)，默认为False，只返回data(即属性)

```python
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data.shape)

data,target = load_iris(return_X_y=True)
print(data.shape)

print(target.shape)

list(iris.target_names)

```

输出依次为：

```
>(150,4)
>(150,4)
>(150,)
>['setosa', 'versicolor', 'virginica']
```



## 2.4 手写数字数据集

### 2.4.1 简介

包含1797个0-9的手写数据，美国由8*8大小矩阵构成，矩阵范围值是0-16，代表颜色深度

### 2.4.2 使用方法

使用sklearn.datasets.load_digits

重要参数：

* return_X_y:表示是否返回target(价格)，默认为False，只返回data(即属性)

* n_class:返回数据类别数，如n_class=5，则返回0-4的数据样本

  ```
  from sklearn.datasets import load_digits
  digits = load_digits()
  print(digits.data.shape)
  print(digits.target.shape)
  print(digits.images.shape)
  
  import matplotlib.pyplot as plt
  plt.matshow(digits.images[1])
  plt.show()
  ```

  输出依次为

  ```
  >(1797, 64)
  >(1797,)
  >(1797, 8, 8)
  ```

￼![image-20200330230314981](C:\Users\kejie\AppData\Roaming\Typora\typora-user-images\image-20200330230314981.png)

## 2.5 sklearn库基本功能

### 2.5.1 分类任务

![image-20200330230559445](C:\Users\kejie\AppData\Roaming\Typora\typora-user-images\image-20200330230559445.png)

### 2.5.2 回归任务

![image-20200330230645360](C:\Users\kejie\AppData\Roaming\Typora\typora-user-images\image-20200330230645360.png)

### 2.5.3 聚类任务

![image-20200330230713865](C:\Users\kejie\AppData\Roaming\Typora\typora-user-images\image-20200330230713865.png)

### 2.5.4 降维任务

![image-20200330230742590](C:\Users\kejie\AppData\Roaming\Typora\typora-user-images\image-20200330230742590.png)



# 7 进一步学习

1. 周志华，机器学习，清华大学出版社
2. Bishop，PRML,Springer
3. Andrew Ng,Machine Learning,http://cs229.stanford.edu/
4. FeiFei Li,CS231n,http://cs231n.stanford.edu/
5. David Silver,Reinforcement Learninng,http://t.cn/RIAfRUt