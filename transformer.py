import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# ===================== 0. 基础配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 任务：数字序列排序（输入乱序，输出升序，完美验证位置/全局依赖）
SEQ_LEN = 10    # 序列长度
VOCAB_SIZE = 15 # 数字0-14
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
D_MODEL = 128   # 模型维度（小模型，快速训练）
N_HEAD = 4
NUM_LAYERS = 3

# ===================== 1. 数据集生成 =====================
def generate_data(batch_size, seq_len, vocab_size):
    """生成乱序输入 + 有序标签的序列数据"""
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(DEVICE)
    y = x.sort(dim=1)[0].to(DEVICE)  # 标签：升序序列
    return x, y

# ===================== 2. 核心模块实现 =====================
# 2.1 位置编码（原文正弦编码 + 可学习编码 + 无编码）
class SinusoidalPositionalEncoding(nn.Module):
    """原文标准正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class LearnablePositionalEncoding(nn.Module):
    """可学习绝对位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# 2.2 多头注意力（标准QKV + 共享KV）
class MultiHeadAttention(nn.Module):
    """标准多头注意力：Q/K/V独立矩阵"""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        # 拆分多头
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        # 注意力计算
        attn_score = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(self.d_k)
        attn_weight = torch.softmax(attn_score, dim=-1)
        output = torch.matmul(attn_weight, v)
        # 拼接多头
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.n_head*self.d_k)
        return self.fc(output)

class SharedKVAttention(nn.Module):
    """消融：共享KV矩阵（K=V）"""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_kv = nn.Linear(d_model, d_model)  # KV共享
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        kv = self.w_kv(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        # K=V
        k = v = kv
        attn_score = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(self.d_k)
        attn_weight = torch.softmax(attn_score, dim=-1)
        output = torch.matmul(attn_weight, v)
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.n_head*self.d_k)
        return self.fc(output)

# 2.3 Transformer层（标准残差 + 无残差 + CNN替代）
class TransformerBlock(nn.Module):
    """标准层：注意力 + 前馈 + 残差 + 层归一化"""
    def __init__(self, d_model, n_head, attn_type='standard'):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head) if attn_type=='standard' else SharedKVAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 残差连接
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class NoResTransformerBlock(nn.Module):
    """消融：无残差连接"""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 移除残差
        attn_out = self.attn(x, x, x)
        x = self.norm1(self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(self.dropout(ffn_out))
        return x

class CNNBlock(nn.Module):
    """消融：CNN替代自注意力"""
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.cnn = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # CNN: [batch, seq_len, d_model] -> [batch, d_model, seq_len]
        x_cnn = x.transpose(1,2)
        x_cnn = self.cnn(x_cnn).transpose(1,2)
        x = self.norm(x + self.dropout(x_cnn))
        x = self.norm(x + self.dropout(self.ffn(x)))
        return x

# ===================== 3. 完整模型定义 =====================
class Transformer(nn.Module):
    """统一模型接口，支持所有消融变体"""
    def __init__(self, vocab_size, d_model, n_head, num_layers, 
                 pos_type='sin', attn_type='standard', res=True, use_attn=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        if pos_type == 'sin':
            self.pos_enc = SinusoidalPositionalEncoding(d_model)
        elif pos_type == 'learnable':
            self.pos_enc = LearnablePositionalEncoding(d_model)
        else:  # 无位置编码
            self.pos_enc = nn.Identity()
        
        # 主干网络
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if not use_attn:
                self.layers.append(CNNBlock(d_model))
            elif not res:
                self.layers.append(NoResTransformerBlock(d_model, n_head))
            else:
                self.layers.append(TransformerBlock(d_model, n_head, attn_type))
        
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)

# ===================== 4. 训练&评估工具 =====================
def train_model(model, name):
    """统一训练函数，返回损失和准确率"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    losses = []
    accs = []
    
    pbar = tqdm(range(EPOCHS), desc=f"训练 {name}")
    for epoch in pbar:
        model.train()
        total_loss = 0
        # 训练
        for _ in range(100):  # 100个batch/epoch
            x, y = generate_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 评估
        model.eval()
        with torch.no_grad():
            x, y = generate_data(1000, SEQ_LEN, VOCAB_SIZE)
            pred = model(x).argmax(dim=-1)
            acc = (pred == y).float().mean().item()
        
        avg_loss = total_loss / 100
        losses.append(avg_loss)
        accs.append(acc)
        pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{acc:.3f}")
    
    return losses, accs

def visualize_results(results):
    """可视化：损失曲线 + 最终准确率对比"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 子图1：训练损失曲线
    plt.figure(figsize=(14, 5))
    plt.subplot(1,2,1)
    for name, (losses, _) in results.items():
        plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('各模型训练损失对比')
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图2：最终准确率
    plt.subplot(1,2,2)
    names = list(results.keys())
    final_accs = [results[name][1][-1] for name in names]
    bars = plt.bar(names, final_accs, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'])
    plt.xlabel('模型变体')
    plt.ylabel('准确率')
    plt.title('各模型最终准确率对比')
    plt.xticks(rotation=45)
    # 标注数值
    for bar, acc in zip(bars, final_accs):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{acc:.3f}", ha='center')
    
    plt.tight_layout()
    plt.show()

# ===================== 5. 主函数：运行所有消融实验 =====================
if __name__ == "__main__":
    # 定义所有消融模型
    models = {
        "1. 标准Transformer(基线)": Transformer(VOCAB_SIZE, D_MODEL, N_HEAD, NUM_LAYERS, pos_type='sin', attn_type='standard', res=True, use_attn=True),
        "2. 无位置编码": Transformer(VOCAB_SIZE, D_MODEL, N_HEAD, NUM_LAYERS, pos_type='none', attn_type='standard', res=True, use_attn=True),
        "3. 可学习位置编码": Transformer(VOCAB_SIZE, D_MODEL, N_HEAD, NUM_LAYERS, pos_type='learnable', attn_type='standard', res=True, use_attn=True),
        "4. 共享KV注意力": Transformer(VOCAB_SIZE, D_MODEL, N_HEAD, NUM_LAYERS, pos_type='sin', attn_type='shared', res=True, use_attn=True),
        "5. 无残差连接": Transformer(VOCAB_SIZE, D_MODEL, N_HEAD, NUM_LAYERS, pos_type='sin', attn_type='standard', res=False, use_attn=True),
        "6. CNN替代自注意力": Transformer(VOCAB_SIZE, D_MODEL, N_HEAD, NUM_LAYERS, pos_type='sin', attn_type='standard', res=True, use_attn=False)
    }

    # 训练所有模型
    results = {}
    print("===== 开始消融实验 =====")
    for name, model in models.items():
        model = model.to(DEVICE)
        losses, accs = train_model(model, name)
        results[name] = (losses, accs)

    # 可视化结果
    print("\n===== 生成可视化图表 =====")
    visualize_results(results)

    # 打印最终结论
    print("\n===== 消融实验核心结论 =====")
    for name, (_, accs) in results.items():
        print(f"{name} | 最终准确率: {accs[-1]:.3f}")
