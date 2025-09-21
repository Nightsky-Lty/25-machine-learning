# EXP2 Report

陈方航 202400300082

## Data

使用 ``numpy`` 加载数据

```python
import numpy as np
import matplotlib.pyplot as plt

x_data = np.loadtxt('ex2x.dat')
y_data = np.loadtxt('ex2y.dat')
```

## Processing your data

为数据添加截距，由于数据的单位不统一，将数据标准化

```python
m = len(x_data)
X = np.c_[np.ones(m), x_data]

std = np.std(X[:,1:],axis = 0)
mean = np.mean(X[:,1:],axis = 0)
X[:,1:] = (X[:,1:] - mean) / std

Y = y_data.reshape((m,1))
n = X.shape[1]
```

## Gradient Descent

预测函数表达式与之前一样:
$$
h_{\theta}=\theta^Tx=\sum_{i=0}^{n}\theta_{i}x_i
$$
梯度下降的更新公式:
$$
\theta_j:=\theta_j-\alpha\dfrac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)} 
$$
将 $\theta$ 的初始值设为 $\vec{0}$

```python
theta = np.zeros((n,1))

def gradient_descent(X, Y, theta, alpha, iterations):
    for _ in range(iterations):
        prediction = X @ theta
        errors = prediction - Y
        gradient = 1.0 / m * (X.T @ errors)

        theta = theta - alpha * gradient
    return theta

```

## Selecting A Learning Rate Using $J(\theta)$

线性回归中，代价函数 $J$ 的表达式:
$$
J(\theta)=\dfrac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2
$$
通过选取不同的 $\alpha$ 画出代价函数随着梯度下降迭代次数上升的收敛图像

```python
def get_J(X, Y, theta):
    prediction = X @ theta
    errors = prediction - Y
    return ((errors.T @ errors) / (2 * m)).item()

alphas = [0.001,0.003,0.01,0.07,0.1,0.5,1]
for alpha in alphas:
    iterations = 50
    J = np.zeros(iterations)
    theta = np.zeros((n,1))
    for i in range(iterations):
        J[i] = get_J(X,Y,theta)
        theta = gradient_descent(X, Y, theta, alpha, 1)
    plt.plot(J)

plt.legend([f'alpha = {alpha}' for alpha in alphas])
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('different alpha')
plt.show()
```

作出图像如下

![](https://raw.githubusercontent.com/Nightsky-Lty/ImageHost/main/20250921195629573.png?token=BNO4ASSRA6WDK4KUXA4EMC3IZ7UCS)

当学习率太小时，需要更多的迭代次数达到收敛，当学习率过大时，$J(\theta)$ 可能没法收敛

最后的 $\theta$ 值是 $340412.65957447,109447.79646964,-6578.35485416$

预测得到的 $Y$ 是 $293081.4643349$

```python
alpha = 0.07
theta = np.zeros((n,1))
theta = gradient_descent(X, Y, theta, alpha, 1650)
print("theta",theta)

predict_X = np.array([[1, 1650, 3]],dtype = float)
predict_X[:,1:] = (predict_X[:,1:] - mean) / std

predict_Y = predict_X @ theta
print("predict_Y:",predict_Y)
```

## Normal Equation

最小二乘拟合的封闭形式如下:
$$
\theta=(X^TX)^{-1}X^{T}\vec{y}
$$
使用这个公式可以不通过循环到 $J(\theta)$ 收敛的方式来得到精确的 $\theta$

```python
def normal_equation(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y

theta_ = normal_equation(X, Y)
print(f"theta_ : {theta}")

predict_Y = predict_X @ theta_
print("predict_Y:",predict_Y)
```

得到的 $\theta$ 和预测值与前文相同