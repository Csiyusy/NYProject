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
    # val_df = df[["datetime", "era5","DireRad", "ScaRad", "temp:C", "humidity:p", "pressure:hPa", "wind_speed:ms", "wind_dir:d", "effective_cloud_cover:p"]]
    val_df = df[["datetime", "era5", "gi_1"]]
    test_df1 = df[["FluxDOWN(W/m²)", "gi_1", "datetime", "atemp", "ahumi", "apress", "ws", "wd", "effective_cloud_cover:p", "Zenith", "Azimuth"]]
    test_df2 = df[["FluxDOWN(W/m²)", "era5", "datetime", "temp:C", "humidity:p", "pressure:hPa", "wind_speed:ms", "wind_speed:ms", "effective_cloud_cover:p", "Zenith", "Azimuth"]]
    test_df1 = preprocess_time_features(test_df1, time_col='datetime')
    test_df2 = preprocess_time_features(test_df2, time_col='datetime')
    num = len(test_df1)

    # 预测并导出结果
    for i in tqdm(range(num - (seq_length + seq_out) * 3), desc='预测结果导出：'):
        input_data1 = test_df1.iloc[i:seq_length * 3 + i:3].values
        input_data2 = test_df2.iloc[i + seq_length * 3:i + (seq_length + seq_out) * 3:3].values
        input_data = np.concatenate([input_data1, input_data2], axis=0)
        if np.isnan(input_data).any():
            continue
        out_data = predict_future(model, input_data, scaler_X, scaler_Y, device).reshape(-1)
        result = val_df.iloc[i + seq_length * 3:i + (seq_length + seq_out) * 3:3]
        result = result.copy()

        result['gi_pre'] = out_data
        result.loc[result['era5'] <= 0, 'gi_pre'] = 0
        # result['era5'] = np.where(result['era5'] > 0, out_data, 0)
        # result = result.rename(columns={
        #     'datetime': '时间', 'era5': '订正总辐射（W/m2）', 'DireRad':'直射辐射（W/m2）','ScaRad':'散射辐射（W/m2）','temp:C': '温度（℃）', 'humidity:p': '湿度（%）',
        #     'pressure:hPa': '压力（hPa）', 'wind_speed:ms': '2米风速（m/s）', 'wind_dir:d': '2米风向（°）',
        #     'effective_cloud_cover:p': '总云量（%）'})

        result.to_csv(os.path.join(out_flod, f'output{i}.csv'), index=False)