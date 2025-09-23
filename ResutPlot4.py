import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
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
data_delte = np.zeros([24,2])
# ======= 生成每个小时的散点图 =======
for i in range(24):
    df = pd.read_csv(input_flod + 'out' + str(i + 1) + '.csv')
    x = df["gi_1"]
    y = df["gi_pre"]
    x2 = df["era5"]

    delta_pre = x - y
    delta_ec = x2 - y

    num = len(x)

    # 创建散点图
    plt.figure(figsize=(8, 6))
    plt.plot(range(num),delta_pre,label="delta_pre", color='r')
    plt.plot(range(num),delta_ec,label="delta_ec", color='b')
    delta1 = np.mean(np.abs(delta_pre))
    delta2 = np.mean(np.abs(delta_ec))

    data_delte[i][0] = delta1
    data_delte[i][1] = delta2

    plt.ylabel("delta")
    plt.legend()
    plt.grid(True)

    plt.savefig(out_flod + f'out{i+1}.png', dpi=300, bbox_inches="tight")
    plt.close()

result1 = np.mean(data_delte[:][0])
result2 = np.mean(data_delte[:][1])
print(result1)
print(result2)

df = pd.DataFrame(data_delte, columns=['pre-obs', 'ec-obs'])
df.to_csv('delte.csv', index=False)