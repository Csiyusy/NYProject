import os
import pandas as pd
import re
from math import sqrt

input_folder = "./result"
output_folder = "./sampled_result"
os.makedirs(output_folder, exist_ok=True)

# 文件名按数字排序
def file_key(fname):
    number = re.findall(r'output(\d+)\.csv', fname)
    return int(number[0]) if number else -1

files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
files.sort(key=file_key)

# 每隔12个采样
sample_files = files[::12]
print(f"按15分钟采样的文件有：{sample_files}")

all_rows = []
for fname in sample_files:
    file_path = os.path.join(input_folder, fname)
    df = pd.read_csv(file_path)
    top4 = df.head(4).copy()
    all_rows.append(top4)

result_df = pd.concat(all_rows, ignore_index=True)

# 对datetime排序
result_df['datetime'] = pd.to_datetime(result_df['datetime'])
result_df.sort_values('datetime', inplace=True)
result_df.to_csv(os.path.join(output_folder, "sampled_result.csv"), index=False)

# ===== 计算 MAE =====
mae_gi_pre_vs_gi_1 = (result_df['gi_pre'] - result_df['gi_1']).abs().mean()
mae_era5_vs_gi_1 = (result_df['era5'] - result_df['gi_1']).abs().mean()

# ===== 计算 RMSE =====
rmse_gi_pre_vs_gi_1 = sqrt(((result_df['gi_pre'] - result_df['gi_1']) ** 2).mean())
rmse_era5_vs_gi_1 = sqrt(((result_df['era5'] - result_df['gi_1']) ** 2).mean())

print(f"MAE(prediction and obs): {mae_gi_pre_vs_gi_1:.4f}")
print(f"MAE(EC and obs): {mae_era5_vs_gi_1:.4f}")
print(f"RMSE(prediction vs obs): {rmse_gi_pre_vs_gi_1:.4f}")
print(f"RMSE(EC vs obs): {rmse_era5_vs_gi_1:.4f}")