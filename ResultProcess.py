import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def calculate_metrics(y_true, y_pred):
    """
    计算 Pearson 相关性和 RMSE。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correlation, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_pred - y_true))

    return correlation, rmse, mae

# 设置数据文件夹路径
folder_path = "./result"       # 输入 CSV 文件的文件夹路径
out_folder = "./24_time_result" # 按时刻输出的新文件夹
os.makedirs(out_folder, exist_ok=True)

# 获取文件列表（只取 .csv 文件）
files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
files.sort() # 按文件名排序，保证遍历稳定
num_files = len(files)
print(f"检测到 {num_files} 个 CSV 文件：", files)

# 创建数据容器：24小时 × 文件数 × 4列
data = np.empty((24, num_files, 4), dtype=object)

# 读取每个文件，将 24小时数据放入容器
for file_idx, file_name in tqdm(enumerate(files), total=num_files, desc='数据整理：'):
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    for hour in range(24):
        data[hour, file_idx, 0] = df['datetime'][hour]
        data[hour, file_idx, 1] = df['era5'][hour]
        data[hour, file_idx, 2] = df['gi_1'][hour]
        data[hour, file_idx, 3] = df['gi_pre'][hour]

# 按小时输出文件并计算指标
for hour in range(24):
    out_path = os.path.join(out_folder, f'out{hour+1}.csv')
    result_hour = data[hour]
    df_hour = pd.DataFrame(result_hour, columns=['datetime', 'era5', 'gi_1', 'gi_pre'])
    # 按时间排序
    df_hour['datetime'] = pd.to_datetime(df_hour['datetime'])  # 转成时间格式
    df_hour.sort_values(by='datetime', inplace=True)
    df_hour.to_csv(out_path, index=False)

    # 计算相关性和RMSE
    test_data = df_hour['gi_1'].values
    pre_data = df_hour['gi_pre'].values
    ec_data = df_hour['era5'].values
    corr, rmse, mae = calculate_metrics(test_data, pre_data)
    corr1, rmse1, mae1 = calculate_metrics(test_data, ec_data)

    print(f'第 {hour+1} 个时刻的结果：')
    print(f"Pearson Correlation: {corr:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    print(f"corr: {corr1:.4f}")
    print(f"RMSE1: {rmse1:.4f}")
    print(f"MAE1: {mae1:.4f}")
