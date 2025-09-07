import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch.optim as optim
import joblib
from tqdm import tqdm
import os


# 编码器
# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        return hn, cn  # 返回隐藏状态


# 解码器
class Decoder(nn.Module):
    def __init__(self, input_dim=128, output_dim=1, hidden_dim=128, num_layers=2, dropout=0.3, future_steps=16):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_steps = future_steps
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.Softplus()

    def forward(self, hn, cn):
        batch_size = hn.size(1)

        # 创建固定长度的输入序列 (batch_size, future_steps, hidden_dim)
        input_seq = torch.zeros(batch_size, self.future_steps, self.hidden_dim, device=hn.device)

        # 通过 LSTM 处理整个未来序列
        out, (hn, cn) = self.lstm(input_seq, (hn, cn))
        out = self.dropout(out)  # Dropout 层

        # 线性层预测 (batch_size, future_steps, output_dim)
        pred = self.fc_out(out)
        pred = self.activate(pred)
        return pred.squeeze(-1)  # 形状变为 (batch_size, future_steps)


# Seq2Seq 结构
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, future_steps=16):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(input_dim=hidden_dim, output_dim=1, hidden_dim=hidden_dim,
                               num_layers=num_layers, dropout=dropout, future_steps=future_steps)

    def forward(self, x):
        hn, cn = self.encoder(x)
        outputs = self.decoder(hn, cn)
        return outputs  # (batch_size, future_steps)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        """
        :param patience: 在多少个 epoch 内，验证损失未改善则停止训练
        :param min_delta: 只有损失减少大于 min_delta 才被认为是改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # 记录最佳参数
        else:
            self.counter += 1
            print(f"Validation loss did not improve for {self.counter}/{self.patience} epochs.")

        if self.counter >= self.patience:
            print("Early stopping triggered.")
            return True  # 训练应停止
        return False  # 继续训练


def preprocess_time_features(df, time_col='time'):
    df = df.copy()  # 防止 SettingWithCopyWarning
    # 时间特征处理
    df[time_col] = pd.to_datetime(df[time_col])  # 转换为时间格式
    df['days_in_year'] = df[time_col].dt.is_leap_year.map(lambda x: 366 if x else 365)  # 366 or 365
    df['day_of_year'] = df[time_col].dt.dayofyear
    df['month'] = df[time_col].dt.month
    df['day_sin'] = np.sin(2 * np.pi * (df['day_of_year'] / df['days_in_year']))  # 归一化时间
    df['day_cos'] = np.cos(2 * np.pi * (df['day_of_year'] / df['days_in_year']))
    df['hour'] = df[time_col].dt.hour
    df['minute'] = df[time_col].dt.minute
    df['time_sin'] = np.sin(2 * np.pi * (df['hour'] * 60 + df['minute']) / 1440)  # 归一化时间
    df['time_cos'] = np.cos(2 * np.pi * (df['hour'] * 60 + df['minute']) / 1440)

    return df.drop(columns=[time_col, 'days_in_year', 'day_of_year', 'minute'])  # 删除原始时间列


def prepare_data(df1, df2, df_y, target_col, predict_time=16, sequence_days=10, interval_per_day=96):
    """
    构造时间序列特征矩阵和目标变量。
    :param df: Pandas DataFrame，包含时间索引和特征列
    :param target_col: 目标变量的列名
    :param sequence_days: 过去多少天的数据作为特征
    :param predict_days: 未来多少天的数据作为目标
    :param interval_per_day: 每天的数据点数（15分钟间隔则为96）
    :return: 处理后的特征矩阵 X 和目标值 y
    """
    sequence_length = sequence_days * interval_per_day

    X, y = [], []
    for i in range(len(df1) - sequence_length * 3 - predict_time * 3):
        x_data1 = df1.iloc[i:i + sequence_length * 3:3].values
        x_data2 = df2.iloc[i + sequence_length * 3:i + sequence_length * 3 + predict_time * 3:3].values
        X.append(np.concatenate([x_data1, x_data2], axis=0))
        y.append(df_y.iloc[i + sequence_length * 3:i + sequence_length * 3 + predict_time * 3:3][target_col].values)

    return np.array(X), np.array(y)


def split_data(X, Y, num_samples, split_ratio=0.8):
    split_index = int(num_samples * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    return X_train, X_test, Y_train, Y_test


def create_train_and_test_data(X, Y, batch_size, feature_dim, num_samples, seq_length, seq_out, split_ratio):
    # 标准化处理
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # 重新调整形状以适配 StandardScaler
    X_reshaped = X.reshape(-1, feature_dim)  # (num_samples * seq_length, feature_dim)

    Y_reshaped = Y.reshape(-1, 1)  # (num_samples * seq_out, 1)

    X_scaled = scaler_X.fit_transform(X_reshaped).reshape(num_samples, seq_length, feature_dim)
    Y_scaled = scaler_y.fit_transform(Y_reshaped).reshape(num_samples, seq_out, 1)

    os.makedirs("./scaler", exist_ok=True)
    joblib.dump(scaler_X, "./scaler/scaler_X_4h.pkl")
    joblib.dump(scaler_y, "./scaler/scaler_Y_4h.pkl")

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

    X_train, X_test, Y_train, Y_test = split_data(X_tensor, Y_tensor, num_samples, split_ratio)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_sampler = SequentialSampler(train_dataset)  # 只打乱batch的顺序，不打乱每个batch内部数据的顺序
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 自定义 Log-Cosh 损失函数
class LogCoshLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true)))


def smoothness_loss(preds, alpha=0.1):
    """
    平滑性正则项：鼓励相邻时间步预测值变化不要太剧烈
    preds: Tensor, shape (batch_size, seq_len)
    """
    diff = preds[:, 1:] - preds[:, :-1]
    return alpha * torch.mean(diff ** 2)


def train_model(train_loader, test_loader, device, model, learning_rate=0.01, num_epochs=50, patience=5):
    # criterion = torch.nn.L1Loss()
    # criterion = LogCoshLoss()
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # 每个 epoch 乘以 0.95
    early_stopping = EarlyStopping(patience=patience)  # 初始化 EarlyStopping

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for batch_X, batch_Y in tepoch:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                optimizer.zero_grad()
                output = model(batch_X)

                loss_main = criterion(output, batch_Y.squeeze(-1))  # MSE 损失
                loss_smooth = smoothness_loss(output, 0.001)
                loss = loss_main + loss_smooth
                loss.backward()
                optimizer.step()  # 这里才是正确的地方，更新参数
                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())  # 在进度条上显示 loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 计算测试损失
        model.eval()
        test_loss = 0
        with torch.no_grad():
            with tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch2:
                for batch_X, batch_Y in tepoch2:
                    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                    output = model(batch_X)
                    loss_main = criterion(output, batch_Y.squeeze(-1))
                    loss = loss_main + loss_smooth
                    test_loss += loss.item()
                    tepoch2.set_postfix(loss=loss.item())  # 在进度条上显示 loss
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        # **在 epoch 级别执行学习率衰减**
        scheduler.step()  # 这里才是正确的位置

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}"
        )

        if early_stopping(test_loss, model):
            print("Stopping early. Restoring best model weights.")
            model.load_state_dict(early_stopping.best_model_state)  # 载入最优模型
            break  # 终止训练

    return train_losses, test_losses


def plot_train_test_loss(train_losses, test_losses, name):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid()
    plt.savefig(name)  # 保存图片


def predict_future(model, data, scaler_x, scaler_y, device):
    data_scaled = scaler_x.transform(data)  # 归一化到 [0,1]

    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_scaled = model(data_tensor)

    predicted_scaled = predicted_scaled.cpu().numpy()

    # 反归一化
    predicted = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()

    return predicted
