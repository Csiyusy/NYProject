import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 读取数据
filename = './24_time_result/out1.csv'
out_flod = './picture/DailyTimeSeries'
file_flod = './24_time_result/'
out_file = './Output_ningyang.csv'
os.makedirs(out_flod, exist_ok=True)

dataset = pd.read_csv(filename)
Time = pd.to_datetime(dataset.iloc[:, 0])
EC_SWF = dataset.iloc[:, 1]
obs_SWF = dataset.iloc[:, 2]

# 初始化 pred_SWF 数组
pred_SWF = np.zeros((len(Time) + 24, 24))

# 读取多个预测文件
for i in range(1, 25):
    filename = file_flod + f'out{i}.csv'
    dataset = pd.read_csv(filename)
    pred_SWF[i - 1:len(Time) + i - 1, i - 1] = dataset.iloc[:, 3]

# 输出结果到 CSV 文件
output = pd.DataFrame({
    'Time': Time,
    'obs_SWF': obs_SWF,
    'EC_SWF': EC_SWF,
    **{f'pred_SWF_{i+1}': pred_SWF[:len(Time), i] for i in range(24)}
})
output.to_csv(out_file, index=False)

# --- 🖼️ 只绘制每天 05:00 ~ 19:00 的图 ---
df_all = output.copy()
df_all['Hour'] = df_all['Time'].dt.hour
df_all['Date'] = df_all['Time'].dt.date

# 按天循环
for date, df_day in df_all.groupby('Date'):
    df_day_filtered = df_day[(df_day['Hour'] >= 5) & (df_day['Hour'] < 19)]

    if df_day_filtered.empty:
        continue

    plt.figure(figsize=(10, 4))
    plt.plot(df_day_filtered['Time'], df_day_filtered['obs_SWF'], '-r', label='Obs')
    plt.plot(df_day_filtered['Time'], df_day_filtered['EC_SWF'], '-k', linewidth=2, label='EC')

    for j in range(24):
        plt.plot(df_day_filtered['Time'], df_day_filtered[f'pred_SWF_{j+1}'], '-.', color=(0.5, 0.5, 0.5))

    plt.title(str(date))
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("SWF")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图像
    plt.savefig(f"{out_flod}/{date}.png")
    plt.close()
