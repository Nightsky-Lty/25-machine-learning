# 实验报告

陈方航 202400300082

## Supervised Learning Problem

读取数据并作出散点图

```python
import numpy as np
import matplotlib.pyplot as plt

x_data = np.loadtxt("ex1x.dat")
y_data = np.loadtxt("ex1y.dat")

plt.xlabel("Age in years")
plt.ylabel("Height in meters")
plt.scatter(x_data,y_data,label = 'Traceing data')
```

作出的图形如下(蓝色部分)：

![](https://cdn.luogu.com.cn/upload/image_hosting/ldnwlgb9.png)

然后添加 $x_0 = 1$ 的截距项

```python
m = len(x_data)
n = 1
X = np.c_[np.ones(m),x_data]
Y = y_data.reshape(m,1)
```

## 2D Linear Regression

线性回归模型表示为
$$
h_{\theta}(x)=\theta^{T}x=\sum_{i=0}^{n}{\theta_ix_i}
$$
批次梯度下降更新公式为
$$
\theta_j:=\theta_j-\alpha\dfrac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$
使用 ``python`` 实现如下

```python
theta = np.zeros((2,1))
alpha = 0.07

def gradient_descent(X, Y, theta, alpha, iterations):
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - Y
        gradient = 1 / m * (X.T @ errors)

        theta = theta - alpha * gradient
    return theta

theta = gradient_descent(X,Y,theta,alpha,1500)
plt.plot(X[:, 1],X @ theta,color = 'orange',label = 'Linear regression')
plt.legend()
plt.show()
```

图像如下

![](https://cdn.luogu.com.cn/upload/image_hosting/ldnwlgb9.png)

使用迭代 $1500$ 次后的 $\theta$ 预测 $3.5$ 与 $7$

```python
predict_x = np.array([3.5,7])
predict_x = np.c_[np.ones(2),predict_x]
predict_y = predict_x @ theta
print("predict_y",predict_y.T)
```

## Understanding $J(\theta)$

根据指南给出的框架作图如下

![](https://cdn.luogu.com.cn/upload/image_hosting/ngiczxlp.png)

![](https://cdn.luogu.com.cn/upload/image_hosting/wceyrq0e.png)

实现方式如下

```python
J_vals = np.zeros((100,100))
theta0_vals = np.linspace(-3,3,100)
theta1_vals = np.linspace(-1,1,100)
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i,j] = np.sum((X @ t - Y) ** 2) / (2 * m)

from mpl_toolkits.mplot3d import Axes3D

theta0_vals_ , theta1_vals_ = np.meshgrid(theta0_vals,theta1_vals)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals_, theta1_vals_, J_vals.T, cmap='turbo')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
plt.title('J')
plt.show()

plt.contourf(theta0_vals_, theta1_vals_, J_vals.T, levels=50, cmap='turbo')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.plot(theta[0], theta[1], 'rx')
plt.legend(['gradient descent'])
plt.show()
```

