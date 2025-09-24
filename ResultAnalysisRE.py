import os
import pandas as pd
import numpy as np

# ===============================
# 计算平均相对误差 (Mean Relative Error)
# ===============================
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = (y_true != 0) & (~np.isnan(y_true))  # 去掉0和NaN
    if np.sum(mask) == 0:
        return 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    re = np.abs((y_pred - y_true) / y_true)  # 相对误差
    filtered = re[re != 1]  # 去掉误差正好为1的
    if len(filtered) == 0:
        result = 0
    else:
        result = np.mean(filtered)
    return result

# ===============================
# 文件路径和结果保存
# ===============================
folder_path = "./result"
result_folder = './Analysis'
os.makedirs(result_folder, exist_ok=True)
result_file = os.path.join(result_folder, 'RE.csv')

# 读取所有CSV文件
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
if not files:
    print("⚠️ 没有找到CSV文件")
else:
    print(f"找到 {len(files)} 个CSV文件：", files)

# 存放计算结果
result_list = []

# 循环读取并计算
for file_name in files:
    file_path = os.path.join(folder_path, file_name)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ 读取文件失败 {file_name}: {e}")
        continue

    if not {'gi_1', 'gi_pre', 'era5'}.issubset(df.columns):
        print(f"⚠️ 文件 {file_name} 缺少必要列")
        continue

    y_true = df["gi_1"]
    y_pred = df["gi_pre"]
    y_era5 = df["era5"]

    re_model = calculate_metrics(y_true, y_pred)   # 模型误差
    re_era5 = calculate_metrics(y_true, y_era5)    # ERA5误差
    diff = re_model - re_era5

    result_list.append([file_name, re_model, re_era5, diff])

    # 显示每个文件的结果
    print(f"{file_name} => 模型RE: {re_model:.4f}, ERA5 RE: {re_era5:.4f}, 差值: {diff:.4f}")

# 保存到CSV
df_result = pd.DataFrame(result_list, columns=['File', 'RE', 'RE_era5', 'diff'])
df_result.to_csv(result_file, index=False)

# 统计信息
print("\n===== 汇总统计 =====")
print('模型平均 RE : ', np.mean(df_result['RE']))
print('ERA5 平均 RE: ', np.mean(df_result['RE_era5']))
num_sum = (df_result['diff'] != 0).sum()
good_data = (df_result['diff'] < 0).sum()
print('比较的文件数: ', num_sum)
print('模型优于 ERA5 的数量: ', good_data)
print('good rate: ', good_data / num_sum if num_sum > 0 else 0)
print(f"结果已保存到 {result_file}")