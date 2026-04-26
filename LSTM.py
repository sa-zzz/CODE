import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import math

# ====================== 1. 加载数据 ======================
df = pd.read_csv("LSTM-Multivariate_pollution.csv")

# 重命名列以匹配代码期望的格式
df.columns = ['date', 'pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'snow', 'rain']

# ====================== 2. 缺失值处理 ======================
df['pm2.5'] = df['pm2.5'].ffill()
df.dropna(inplace=True)

# ====================== 3. 构造日期索引 ======================
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# ====================== 4. 分类特征编码（风向 cbwd） ======================
le = LabelEncoder()
df['cbwd'] = le.fit_transform(df['cbwd'])

# ====================== 5. 数据归一化 ======================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# ====================== 6. 构建时间序列样本 ======================
# 用前 look_back 小时预测下1小时 PM2.5
look_back = 24  # 使用过去1天数据
n_features = scaled_data.shape[1]

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, :])   # 所有特征
        y.append(data[i, 0])               # 预测 pm2.5（第0列）
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, look_back)

# 划分训练集（80%）
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM 输入形状：[样本数, 时间步, 特征数]
print("训练集形状:", X_train.shape)

# ====================== 7. 搭建 LSTM 模型 ======================
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ====================== 8. 训练 ======================
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1
)

# ====================== 9. 预测 ======================
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# ====================== 10. 反归一化 ======================
# 构造空数组以匹配 scaler 维度
train_pred_seq = np.zeros(shape=(len(train_pred), n_features))
train_pred_seq[:,0] = train_pred[:,0]
train_pred = scaler.inverse_transform(train_pred_seq)[:,0]

y_train_seq = np.zeros(shape=(len(y_train), n_features))
y_train_seq[:,0] = y_train
y_train_ori = scaler.inverse_transform(y_train_seq)[:,0]

test_pred_seq = np.zeros(shape=(len(test_pred), n_features))
test_pred_seq[:,0] = test_pred[:,0]
test_pred = scaler.inverse_transform(test_pred_seq)[:,0]

y_test_seq = np.zeros(shape=(len(y_test), n_features))
y_test_seq[:,0] = y_test
y_test_ori = scaler.inverse_transform(y_test_seq)[:,0]

# ====================== 11. 计算指标 ======================
train_rmse = math.sqrt(mean_squared_error(y_train_ori, train_pred))
test_rmse = math.sqrt(mean_squared_error(y_test_ori, test_pred))
train_mae = mean_absolute_error(y_train_ori, train_pred)
test_mae = mean_absolute_error(y_test_ori, test_pred)

print(f'训练集 RMSE: {train_rmse:.2f}')
print(f'测试集 RMSE: {test_rmse:.2f}')
print(f'训练集 MAE: {train_mae:.2f}')
print(f'测试集 MAE: {test_mae:.2f}')

# ====================== 12. 绘图 ======================
plt.figure(figsize=(16,8))
plt.title('LSTM PM2.5 多变量预测')
plt.plot(y_test_ori, label='真实 PM2.5')
plt.plot(test_pred, label='预测 PM2.5')
plt.xlabel('时间步')
plt.ylabel('PM2.5 浓度')
plt.legend()
plt.show()

# 绘制 loss 曲线
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()