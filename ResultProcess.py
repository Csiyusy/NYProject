import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def calculate_metrics(y_true, y_pred):
    """
    计算 Pearson 相关性和 RMSE。

    :param y_true: 真实值 (NumPy 数组)
    :param y_pred: 预测值 (NumPy 数组)
    :return: Pearson 相关系数, RMSE
    """
    # 确保输入是 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算 Pearson 相关系数
    correlation, _ = pearsonr(y_true, y_pred)

    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return correlation, rmse


# 设置数据文件夹路径
folder_path = "./result"  # 修改为你的文件夹路径
out_flod = "./24_time_result"

os.makedirs(out_flod, exist_ok=True)

files = os.listdir(folder_path)
num = len(files)
data = np.empty((24, num, 4), dtype=object)
for i in tqdm(range(num), desc='数据整理：'):
    file_path = folder_path + '/' + 'output' + str(i) + '.csv'
    df = pd.read_csv(file_path)
    for j in range(24):
        data[j, i, 0] = df['datetime'][j]
        data[j, i, 1] = df['era5'][j]
        data[j, i, 2] = df['gi_1'][j]
        data[j, i, 3] = df['gi_pre'][j]

for i in range(24):
    out_path = out_flod + '/' + 'out' + str(i + 1) + '.csv'
    result = data[i]
    df_result = pd.DataFrame(result, columns=['datetime', 'era5', 'gi_1', 'gi_pre'])
    df_result.to_csv(out_path, index=False)

    test_data = df_result['gi_1'].values
    pre_data = df_result['gi_pre'].values
    corr, rmse = calculate_metrics(test_data, pre_data)
    print('第'+str(i+1)+'个时刻的结果：')
    print(f"Pearson Correlation: {corr:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")



