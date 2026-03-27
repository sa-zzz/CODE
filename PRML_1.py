import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression

# 绘图中文设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================
# 1. 读取数据（直接放在同文件夹即可运行）
# ==============================================
train_df = pd.read_excel("Data4Regression.xlsx", sheet_name=0)
test_df = pd.read_excel("Data4Regression.xlsx", sheet_name=1)

print("="*50)
print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)

# 提取特征与标签
X_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y_complex'].values
X_test = test_df['x_new'].values.reshape(-1, 1)
y_test = test_df['y_new_complex'].values

# 添加偏置项（用于线性模型）
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# ==============================================
# 2. 绘制原始数据分布
# ==============================================
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.scatter(X_train, y_train, c='blue', s=15)
plt.title("训练数据分布")
plt.grid(alpha=0.3)

plt.subplot(122)
plt.scatter(X_test, y_test, c='purple', s=15)
plt.title("测试数据分布")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================================
# 3. 线性拟合：最小二乘法
# ==============================================
def least_squares(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

w_ls = least_squares(X_train_b, y_train)
y_ls_train = X_train_b @ w_ls
y_ls_test = X_test_b @ w_ls

# ==============================================
# 4. 线性拟合：梯度下降法 GD
# ==============================================
def gradient_descent(X, y, lr=0.01, iter=2000):
    w = np.random.randn(2)
    m = len(X)
    for _ in range(iter):
        grad = 2/m * X.T @ (X @ w - y)
        w -= lr * grad
    return w

w_gd = gradient_descent(X_train_b, y_train)
y_gd_train = X_train_b @ w_gd
y_gd_test = X_test_b @ w_gd

# ==============================================
# 5. 线性拟合：牛顿法
# ==============================================
def newton_method(X, y):
    w = np.random.randn(2).reshape(-1,1)
    y = y.reshape(-1,1)
    H = 2/len(X) * X.T @ X
    grad = 2/len(X) * X.T @ (X @ w - y)
    w -= np.linalg.inv(H) @ grad
    return w.ravel()

w_nt = newton_method(X_train_b, y_train)
y_nt_train = X_train_b @ w_nt
y_nt_test = X_test_b @ w_nt

# ==============================================
# 6. 输出线性模型误差
# ==============================================
def show_error(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} | MSE: {mse:.4f} | R²: {r2:.4f}")

print("\n" + "="*30 + " 线性拟合结果 " + "="*30)
show_error(y_train, y_ls_train, "最小二乘 训练集")
show_error(y_test, y_ls_test, "最小二乘 测试集")
show_error(y_train, y_gd_train, "梯度下降 训练集")
show_error(y_test, y_gd_test, "梯度下降 测试集")
show_error(y_train, y_nt_train, "牛顿法   训练集")
show_error(y_test, y_nt_test, "牛顿法   测试集")

# ==============================================
# 7. 绘制线性拟合图
# ==============================================
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.scatter(X_train, y_train, s=10, c='blue')
plt.plot(X_train, y_ls_train, 'r-', label='最小二乘')
plt.plot(X_train, y_gd_train, 'g--', label='梯度下降')
plt.plot(X_train, y_nt_train, '-.', color='orange', label='牛顿法')
plt.title("线性拟合 - 训练集")
plt.legend()

plt.subplot(122)
plt.scatter(X_test, y_test, s=10, c='purple')
plt.plot(X_test, y_ls_test, 'r-', label='最小二乘')
plt.plot(X_test, y_gd_test, 'g--', label='梯度下降')
plt.plot(X_test, y_nt_test, '-.', color='orange', label='牛顿法')
plt.title("线性拟合 - 测试集")
plt.legend()
plt.show()

# ==============================================
# 8. 非线性拟合：多项式回归（最优阶数自动选择）
# ==============================================
max_deg = 10
cv_scores = []
for d in range(1, max_deg+1):
    poly = PolynomialFeatures(degree=d)
    X_p = poly.fit_transform(X_train)
    score = cross_val_score(LinearRegression(), X_p, y_train, cv=5).mean()
    cv_scores.append(score)

best_deg = np.argmax(cv_scores) + 1
print(f"\n最优多项式阶数: {best_deg}")

# 最优阶拟合
poly_best = PolynomialFeatures(degree=best_deg)
X_train_p = poly_best.fit_transform(X_train)
X_test_p = poly_best.transform(X_test)
model = LinearRegression()
model.fit(X_train_p, y_train)
y_p_train = model.predict(X_train_p)
y_p_test = model.predict(X_test_p)

# ==============================================
# 9. 输出非线性拟合结果
# ==============================================
print("\n" + "="*30 + f" 非线性拟合（{best_deg}阶） " + "="*30)
show_error(y_train, y_p_train, "训练集")
show_error(y_test, y_p_test, "测试集")

# ==============================================
# 10. 绘制非线性拟合曲线
# ==============================================
x_line = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1,1)
x_line_p = poly_best.transform(x_line)
y_line = model.predict(x_line_p)

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.scatter(X_train, y_train, s=12, c='blue')
plt.plot(x_line, y_line, 'r-', linewidth=2)
plt.title(f"训练集 - {best_deg}阶多项式拟合")

plt.subplot(122)
plt.scatter(X_test, y_test, s=12, c='purple')
plt.plot(x_line, y_line, 'r-', linewidth=2)
plt.title(f"测试集 - {best_deg}阶多项式拟合")
plt.show()
