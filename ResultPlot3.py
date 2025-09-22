import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

input_flod = './24_time_result/'
out_flod = './picture/ScatterPlot/'
os.makedirs(out_flod, exist_ok=True)
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
    # rmse = mean_squared_error(x, y, squared=False)  # squared=False 表示直接取 RMSE
    mae = mean_absolute_error(x, y)

    # 创建散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, label="Data points")

    # 生成 x 值范围
    x_vals = np.linspace(x.min(), x.max(), 100)

    # 绘制 1:1 线
    plt.plot(x_vals, x_vals, 'r', label="1:1 Line")

    # 绘制相对误差区间线 y = 1.2x 和 y = 0.8x
    plt.plot(x_vals, 1.2 * x_vals, 'g--', label="Upper Bound (1.2x)")
    plt.plot(x_vals, 0.8 * x_vals, 'g--', label="Lower Bound (0.8x)")

    # 在图上显示落在误差区间内的占比和指标
    text_x = x.min() + (x.max() - x.min()) * 0.05
    text_y = y.min() + (y.max() - y.min()) * 0.7

    plt.text(text_x, text_y, f"Within Bound: {percentage_within_bounds:.2f}%", fontsize=11, color="g")
    plt.text(text_x, text_y - (y.max() - y.min()) * 0.08, f"R²: {r2:.3f}", fontsize=11)
    plt.text(text_x, text_y - (y.max() - y.min()) * 0.16, f"RMSE: {rmse:.3f}", fontsize=11)
    plt.text(text_x, text_y - (y.max() - y.min()) * 0.24, f"MAE: {mae:.3f}", fontsize=11)

    # 添加标签和图例
    plt.xlabel("global_rad:W")
    plt.ylabel("predict")
    plt.title("LSTM Scatter Plot with 1:1 Line and Relative Error Bounds (80%)")
    plt.legend()
    plt.grid(True)

    # 保存图像
    image_path = out_flod + 'out' + str(i + 1) + '.png'
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()

# 读取所有 CSV 合并
all_df_list = []
for i in range(24):
    df = pd.read_csv(input_flod + f'out{i + 1}.csv')
    all_df_list.append(df)

# 合并后去重，防止重复值影响统计结果
all_df = pd.concat(all_df_list, ignore_index=True)

if "datetime" in all_df.columns:
    all_df = all_df.drop_duplicates(subset=["datetime"], keep='first')
else:
    all_df = all_df.drop_duplicates()

# 取 era5 和 gi_1
x_era5 = all_df["gi_1"]
y_era5 = all_df["era5"]  # 确保 CSV 中有这一列

# 计算指标
within_bounds_era5 = ((y_era5 >= 0.8 * x_era5) & (y_era5 <= 1.2 * x_era5)).sum()
percentage_within_bounds_era5 = within_bounds_era5 / len(x_era5) * 100

r2_era5 = r2_score(x_era5, y_era5)
rmse_era5 = np.sqrt(mean_squared_error(x_era5, y_era5))
mae_era5 = mean_absolute_error(x_era5, y_era5)

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(x_era5, y_era5, alpha=0.5, label="ERA5")

x_vals_era5 = np.linspace(x_era5.min(), x_era5.max(), 100)
plt.plot(x_vals_era5, x_vals_era5, 'r', label="1:1 Line")
plt.plot(x_vals_era5, 1.2 * x_vals_era5, 'g--', label="Upper Bound (1.2x)")
plt.plot(x_vals_era5, 0.8 * x_vals_era5, 'g--', label="Lower Bound (0.8x)")

text_x_era5 = x_era5.min() + (x_era5.max() - x_era5.min()) * 0.05
text_y_era5 = y_era5.min() + (y_era5.max() - y_era5.min()) * 0.7

plt.text(text_x_era5, text_y_era5, f"Within Bound: {percentage_within_bounds_era5:.2f}%", fontsize=11, color="g")
plt.text(text_x_era5, text_y_era5 - (y_era5.max() - y_era5.min()) * 0.08, f"R²: {r2_era5:.3f}", fontsize=11)
plt.text(text_x_era5, text_y_era5 - (y_era5.max() - y_era5.min()) * 0.16, f"RMSE: {rmse_era5:.3f}", fontsize=11)
plt.text(text_x_era5, text_y_era5 - (y_era5.max() - y_era5.min()) * 0.24, f"MAE: {mae_era5:.3f}", fontsize=11)

plt.xlabel("global_rad:W (GI_1)")
plt.ylabel("ERA5")
plt.title("ERA5 Scatter Plot")
plt.legend()
plt.grid(True)

plt.savefig(out_flod + 'ERA5.png', dpi=300, bbox_inches="tight")
plt.close()

print("✅ 所有散点图已保存完毕！")