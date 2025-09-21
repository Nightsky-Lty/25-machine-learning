import numpy as np
import matplotlib.pyplot as plt

x_data = np.loadtxt('ex2x.dat')
y_data = np.loadtxt('ex2y.dat')

m = len(x_data)
X = np.c_[np.ones(m), x_data]

std = np.std(X[:,1:],axis = 0)
mean = np.mean(X[:,1:],axis = 0)
X[:,1:] = (X[:,1:] - mean) / std

Y = y_data.reshape((m,1))
n = X.shape[1]
theta = np.zeros((n,1))
alpha = 0.07

def gradient_descent(X, Y, theta, alpha, iterations):
    for _ in range(iterations):
        prediction = X @ theta
        errors = prediction - Y
        gradient = 1.0 / m * (X.T @ errors)

        theta = theta - alpha * gradient
    return theta

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

alpha = 0.07
theta = np.zeros((n,1))
theta = gradient_descent(X, Y, theta, alpha, 1650)
print("theta",theta)

predict_X = np.array([[1, 1650, 3]],dtype = float)
predict_X[:,1:] = (predict_X[:,1:] - mean) / std

predict_Y = predict_X @ theta
print("predict_Y:",predict_Y)

def normal_equation(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y

theta_ = normal_equation(X, Y)
print(f"theta_ : {theta}")

predict_Y = predict_X @ theta_
print("predict_Y:",predict_Y)