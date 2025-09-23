import matplotlib.pyplot as plt
import pandas as pd
import os

input_flod = './24_time_result/'
out_flod = './picture/LineChart/'

os.makedirs(out_flod, exist_ok=True)

# 为每个月绘制单独的折线图
for i in range(24):
    df = pd.read_csv(input_flod + 'out'+str(i+1)+'.csv')

    df['datetime'] = pd.to_datetime(df['datetime'])
    df["month"] = df["datetime"].dt.month
    unique_months = df["month"].unique()
    for month in unique_months:
        df_month = df[df["month"] == month]

        plt.figure(figsize=(12, 6))
        plt.plot(df_month["datetime"], df_month["era5"], label="EC", linestyle="-", color="blue")
        plt.plot(df_month["datetime"], df_month["gi_1"], label="obs", linestyle="--", color="red")
        plt.plot(df_month["datetime"], df_month["gi_pre"], label="prediction", linestyle="-.", color="green")

        plt.xlabel("Datetime")
        plt.ylabel("Values")
        plt.title(f"Time Series of FluxDOWN, global_rad, and predict ({month})")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid()

        image_path = out_flod + 'out'+str(i+1)+'_'+str(month)+'.png'
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close()