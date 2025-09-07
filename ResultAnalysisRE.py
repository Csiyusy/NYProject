import os
import pandas as pd
import numpy as np


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = (y_true != 0) & (~np.isnan(y_true))
    if np.sum(mask) == 0:
        return 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    re = np.abs((y_pred - y_true) / y_true)
    filtered = re[re != 1]
    if len(filtered) == 0:
        result = 0
    else:
        result = np.mean(filtered)
    return result


folder_path = "./result"
result_file = './Analysis'
os.makedirs(result_file, exist_ok=True)
result_file = os.path.join(result_file, 'RE.csv')

files = os.listdir(folder_path)
num = len(files)
result_list = []

for i in range(num):
    file_path = f'{folder_path}/output{i}.csv'
    if not os.path.exists(file_path):
        print(f'File not exists: {file_path}')
        continue
    df = pd.read_csv(file_path)
    y_true = df["gi_1"]
    y_pred = df["gi_pre"]
    y_era5 = df["era5"]

    re = calculate_metrics(y_true, y_pred)
    re2 = calculate_metrics(y_true, y_era5)
    dif = re - re2
    result_list.append([re, re2, dif])

df_result = pd.DataFrame(result_list, columns=['RE', 'RE_era5', 'diff'])
df_result.to_csv(result_file, index=False)
print('pre result: ', np.mean(df_result['RE']))
print('era5 result: ', np.mean(df_result['RE_era5']))
num_sum = (df_result['diff'] != 0).sum()
good_data = (df_result['diff'] < 0).sum()
print(num_sum)
print(good_data)
print('good rate: ', good_data / num_sum)