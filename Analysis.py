import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# 设置数据文件夹路径
input_flod = './24_time_result/'

def relative_error_np(y_true, y_pred, epsilon=1e-8):
    """
    计算 NumPy 格式的相对误差。
    参数:
        y_true: 真实值 (np.ndarray)
        y_pred: 预测值 (np.ndarray)
        epsilon: 防止除零的小数
    返回:
        相对误差数组
    """
    return np.abs(y_pred - y_true) / (np.abs(y_true) + epsilon)

for i in range(24):
    df = pd.read_csv(input_flod + 'out'+str(i+1)+'.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])  # 转为 datetime 类型
    # 添加一个日期列（只保留年月日）
    df['date'] = df['datetime'].dt.date
    # 按日期分组并求和
    daily_sum = df.groupby('date').sum(numeric_only=True)

    pre = daily_sum['gi_pre']
    era5 = daily_sum['era5']
    true = daily_sum['gi_1']
    sum_pre = np.sum(pre)
    sum_era5 = np.sum(era5)
    sum_true = np.sum(true)
    print("pre:")
    print(1-relative_error_np(sum_true,sum_pre))
    print("era5:")
    print(1-relative_error_np(sum_true, sum_era5))