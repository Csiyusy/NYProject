import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

input_flod = './24_time_result/'
out_flod = './picture/ScatterPlot/'
os.makedirs(out_flod, exist_ok=True)

# 用于存储每个文件的 RMSE、R²、MAE
rmse_list = []
r2_list = []
mae_list = []
percentage_list = []

# ======= 生成每个小时的散点图 =======
for i in range(24):
    df = pd.read_csv(input_flod + 'out' + str(i + 1) + '.csv')
    x = df["gi_1"]
    y = df["gi_pre"]

    # 计算落在相对误差区间 [0.8x, 1.2x] 的点数
    within_bounds = ((y >= 0.8 * x) & (y <= 1.2 * x)).sum()
    total_points = len(x)
    percentage_within_bounds = (within_bounds / total_points) * 100

    # 计算 R²、RMSE、MAE
    r2 = r2_score(x, y)
    mse = mean_squared_error(x, y)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(x, y)

    # 保存指标
    rmse_list.append(rmse)
    r2_list.append(r2)
    mae_list.append(mae)
    percentage_list.append(percentage_within_bounds)

    # 创建散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, label="Data points")
    x_vals = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_vals, x_vals, 'r', label="1:1 Line")
    plt.plot(x_vals, 1.2 * x_vals, 'g--', label="Upper Bound (1.2x)")
    plt.plot(x_vals, 0.8 * x_vals, 'g--', label="Lower Bound (0.8x)")

    # 指标文本
    text_x = x.min() + (x.max() - x.min()) * 0.05
    text_y = y.min() + (y.max() - y.min()) * 0.7
    plt.text(text_x, text_y, f"Within Bound: {percentage_within_bounds:.2f}%", fontsize=11, color="g")
    plt.text(text_x, text_y - (y.max() - y.min()) * 0.08, f"R²: {r2:.3f}", fontsize=11)
    plt.text(text_x, text_y - (y.max() - y.min()) * 0.16, f"RMSE: {rmse:.3f}", fontsize=11)
    plt.text(text_x, text_y - (y.max() - y.min()) * 0.24, f"MAE: {mae:.3f}", fontsize=11)

    plt.xlabel("global_rad:W")
    plt.ylabel("predict")
    plt.title(f"LSTM Scatter Plot (Hour {i+1})")
    plt.legend()
    plt.grid(True)

    plt.savefig(out_flod + f'out{i+1}.png', dpi=300, bbox_inches="tight")
    plt.close()

# ======= 绘制 RMSE 曲线/柱状图 =======
hours = list(range(1, 25))

plt.figure(figsize=(10, 6))
plt.bar(hours, rmse_list, color='skyblue')
plt.xlabel("Hour")
plt.ylabel("RMSE")
plt.title("RMSE per Hour")
plt.xticks(hours)
for h, r in zip(hours, rmse_list):
    plt.text(h, r + 0.01, f"{r:.2f}", ha='center', fontsize=9)  # 显示数值
plt.grid(axis='y')

plt.savefig(out_flod + 'RMSE_per_hour.png', dpi=300, bbox_inches="tight")
plt.close()

print("✅ 散点图和 RMSE 图已保存完毕！")