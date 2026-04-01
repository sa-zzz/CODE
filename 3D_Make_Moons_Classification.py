import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ===================== 1. 3D月牙数据生成函数（你的代码） =====================
def make_moons_3d(n_samples_per_class=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples_per_class)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    # 生成两类数据：C0(0) 和 C1(1)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])

    # 添加高斯噪声
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

# ===================== 2. 生成训练集 + 测试集 =====================
# 训练集：1000个数据（500 C0 + 500 C1）
X_train, y_train = make_moons_3d(n_samples_per_class=500, noise=0.2)
# 测试集：500个数据（250 C0 + 250 C1）→ 同分布独立数据
X_test, y_test = make_moons_3d(n_samples_per_class=250, noise=0.2)

# 可视化3D训练数据
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], 
                     c=y_train, cmap='viridis', s=20, alpha=0.7)
ax.legend(*scatter.legend_elements(), title="Classes")
ax.set_xlabel('X Axis', fontsize=12)
ax.set_ylabel('Y Axis', fontsize=12)
ax.set_zlabel('Z Axis', fontsize=12)
plt.title('3D Make Moons - Training Dataset (1000 samples)', fontsize=14)
plt.show()

# ===================== 3. 定义并训练所有模型 =====================
# 1. 单决策树
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# 2. AdaBoost + 决策树（基评估器为决策树）
adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),
                              n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)

# 3. SVM 三种核函数
svm_linear = SVC(kernel='linear', random_state=42)  # 线性核
svm_rbf = SVC(kernel='rbf', random_state=42)        # 高斯核（默认）
svm_poly = SVC(kernel='poly', degree=3, random_state=42) # 多项式核

svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)
svm_poly.fit(X_train, y_train)

# ===================== 4. 模型测试与性能评估 =====================
models = {
    "Decision Tree": dt,
    "AdaBoost + Decision Tree": adaboost,
    "SVM (Linear Kernel)": svm_linear,
    "SVM (RBF Kernel)": svm_rbf,
    "SVM (Polynomial Kernel)": svm_poly
}

print("="*60)
print("3D月牙数据集 分类模型测试结果 (测试集500个样本)")
print("="*60)
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}:")
    print(f"测试集准确率 = {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["C0", "C1"], digits=3))

# ===================== 5. 测试集3D可视化 =====================
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], 
                     c=y_test, cmap='viridis', s=30, alpha=0.8)
ax.legend(*scatter.legend_elements(), title="Classes")
ax.set_xlabel('X Axis', fontsize=12)
ax.set_ylabel('Y Axis', fontsize=12)
ax.set_zlabel('Z Axis', fontsize=12)
plt.title('3D Make Moons - Test Dataset (500 samples)', fontsize=14)
plt.show()