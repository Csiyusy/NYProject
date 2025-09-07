from src.model import *

def train_lstm_model(
    train_csv_path,
    old_model_name,
    save_model_path,
    loss_png_path,
    batch_size=64,
    split_ratio=0.8,
    feature_dim=16,
    seq_length=960 + 96 * 3,
    seq_out=96 * 3,
    num_epochs=30,
    learning_rate=0.001
):
    # 数据读取和预处理
    df = pd.read_csv(train_csv_path)
    df_y = df[["gi_1"]]
    df1 = df[
        ["FluxDOWN(W/m²)", "gi_1", "datetime", "atemp", "ahumi", "apress", "ws", "wd",
         "effective_cloud_cover:p", "Zenith", "Azimuth"]]
    df2 = df[
        ["FluxDOWN(W/m²)", "era5", "datetime", "temp:C", "humidity:p", "pressure:hPa",
         "wind_speed:ms", "wind_speed:ms", "effective_cloud_cover:p", "Zenith", "Azimuth"]]
    df1 = preprocess_time_features(df1, time_col='datetime')
    df2 = preprocess_time_features(df2, time_col='datetime')

    X, y = prepare_data(df1, df2, df_y, target_col='gi_1', predict_time=seq_out)
    valid_x = ~np.isnan(X).any(axis=(1, 2))
    valid_y = ~np.isnan(y).any(axis=1)
    valid_idx = valid_x & valid_y
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]
    num_samples = len(X_clean)

    # 数据加载器
    train_loader, test_loader = create_train_and_test_data(
        X_clean, y_clean, batch_size, feature_dim, num_samples, seq_length, seq_out, split_ratio
    )

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(feature_dim, future_steps=seq_out)

    if old_model_name and os.path.exists(old_model_name):
        print(f"发现旧模型：{old_model_name}，正在加载权重。")
        model.load_state_dict(torch.load(old_model_name, map_location=device))
    else:
        print(f"未发现旧模型 {old_model_name}，将使用初始权重训练。")

    model.to(device)

    # 训练
    train_losses, test_losses = train_model(
        train_loader, test_loader, device, model, learning_rate, num_epochs
    )

    # 可视化和保存模型
    plot_train_test_loss(train_losses, test_losses, loss_png_path)
    torch.save(model.state_dict(), save_model_path)
    print("模型已保存！")