from src.model import *

def run_predict(
    test_file,
    model_name,
    out_flod,
    scaler_X_path,
    scaler_Y_path,
    seq_length=960,
    seq_out=16,
    input_dim=16
):
    # 创建输出文件夹
    os.makedirs(out_flod, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = Seq2Seq(input_dim=input_dim,future_steps=seq_out)
    model.load_state_dict(torch.load(model_name))
    model.to(device)

    # 加载scaler
    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)

    # 加载数据
    df = pd.read_csv(test_file)
    # 输出结果版
    # val_df = df[["datetime", "era5","DireRad", "ScaRad", "temp:C", "humidity:p", "pressure:hPa", "wind_speed:ms", "wind_dir:d", "effective_cloud_cover:p"]]
    # 画图版
    val_df = df[["datetime", "era5", "gi_1"]]
    test_df1 = df[["FluxDOWN(W/m²)", "gi_1", "datetime", "atemp", "ahumi", "apress", "ws", "wd", "effective_cloud_cover:p", "Zenith", "Azimuth"]]
    test_df2 = df[["FluxDOWN(W/m²)", "era5", "datetime", "temp:C", "humidity:p", "pressure:hPa", "wind_speed:ms", "wind_speed:ms", "effective_cloud_cover:p", "Zenith", "Azimuth"]]
    time_series = pd.to_datetime(test_df1["datetime"])
    test_df1 = preprocess_time_features(test_df1, time_col='datetime')
    test_df2 = preprocess_time_features(test_df2, time_col='datetime')
    num = len(test_df1)

    first_idx = None
    for idx in range(0, num - (seq_length + seq_out) * 3):
        t = time_series.iloc[idx]
        if t.minute == 0 and t.hour in [0, 6, 12, 18]:
            first_idx = idx
            break

    if first_idx is None:
        raise ValueError("找不到符合 0:00/6:00/12:00/18:00 的起点")

    slide_step = 1  # 每次滑动的距离 1 = 5min
    # 预测并导出结果
    for i in tqdm(range(first_idx, num - (seq_length + seq_out) * 3,slide_step), desc='预测结果导出：'):
        input_data1 = test_df1.iloc[i:seq_length * 3 + i:3].values
        input_data2 = test_df2.iloc[i + seq_length * 3:i + (seq_length + seq_out) * 3:3].values
        input_data = np.concatenate([input_data1, input_data2], axis=0)
        if np.isnan(input_data).any():
            continue
        out_data = predict_future(model, input_data, scaler_X, scaler_Y, device).reshape(-1)
        # 保留小数点后两位
        out_data = np.round(out_data,2)

        result = val_df.iloc[i + seq_length * 3:i + (seq_length + seq_out) * 3:3]
        result = result.copy()

        # 画图版
        result['gi_pre'] = out_data
        result.loc[result['era5'] <= 0, 'gi_pre'] = 0
        result.to_csv(os.path.join(out_flod, f'output{i}.csv'), index=False)

        # 输出结果版
        # result['era5'] = np.where(result['era5'] > 0, out_data, 0)
        # result = result.rename(columns={
        #     'datetime': '时间', 'era5': '订正总辐射（W/m2）', 'DireRad':'直射辐射（W/m2）','ScaRad':'散射辐射（W/m2）','temp:C': '温度（℃）', 'humidity:p': '湿度（%）',
        #     'pressure:hPa': '压力（hPa）', 'wind_speed:ms': '2米风速（m/s）', 'wind_dir:d': '2米风向（°）',
        #     'effective_cloud_cover:p': '总云量（%）'})
        #
        # # 生成有意义的文件名
        # start_time = pd.to_datetime(result.iloc[0]['时间']).strftime("%Y%m%d_%H%M")
        # end_time = pd.to_datetime(result.iloc[-1]['时间']).strftime("%Y%m%d_%H%M")
        # filename = f"{start_time}-{end_time}_pre.csv"
        #
        # # 保存
        # result.to_csv(os.path.join(out_flod, filename), index=False)