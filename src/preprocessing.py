import pandas as pd
import os
import numpy as np

# 合并多个原始CSV
def merge_raw_csv(file1_path, file2_path, file_flod, merged_file_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    df1['datetime'] = pd.to_datetime(df1['datetime'])
    df2['datetime'] = pd.to_datetime(df2['datetime'])
    merged_df = pd.merge(df1, df2, on='datetime', how='inner')
    merged_df['air_impact'] = merged_df['FluxDOWN(W/m²)'] - merged_df['gi_1']
    merged_df.loc[merged_df['air_impact'] < 0, 'air_impact'] = 0
    merged_df.loc[merged_df['FluxDOWN(W/m²)'] < 0.01, 'FluxDOWN(W/m²)'] = 0
    # 完整时间序列
    full_time_range = pd.DataFrame({'datetime': pd.date_range(start=merged_df['datetime'].min(),
                                                              end=merged_df['datetime'].max(),
                                                              freq='5min')})
    dataset = pd.merge(full_time_range, merged_df, on='datetime', how='left')
    all_weather_data = []
    files = [f for f in os.listdir(file_flod) if f.lower().endswith('.csv')]
    for file in files:
        file_path = os.path.join(file_flod, file)
        df = pd.read_csv(file_path)[
            ['时间', '总辐射（W/m2）', '直射辐射（W/m2）', '散射辐射（W/m2）', '温度（℃）', '湿度（%）', '压力（hPa）',
             '2米风速（m/s）', '2米风向（°）','总云量（%）']]
        all_weather_data.append(df)
    weather_data = pd.concat(all_weather_data).groupby('时间', group_keys=False).last()
    weather_data = weather_data.reset_index()
    weather_data = weather_data.rename(columns={
        '时间': 'datetime','总辐射（W/m2）': 'era5', '直射辐射（W/m2）': 'DireRad','散射辐射（W/m2）':'ScaRad','温度（℃）': 'temp:C', '湿度（%）': 'humidity:p',
        '压力（hPa）': 'pressure:hPa', '2米风速（m/s）': 'wind_speed:ms', '2米风向（°）': 'wind_dir:d',
        '总云量（%）': 'effective_cloud_cover:p'})
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
    # 插值为5分钟
    weather_data = weather_data.set_index('datetime')
    weather_data_interp = weather_data.resample('5min').interpolate(method='time').reset_index()
    # 合并
    df_merged = pd.merge(dataset, weather_data_interp, on='datetime', how='left')
    os.makedirs(os.path.dirname(merged_file_path), exist_ok=True)
    df_merged.to_csv(merged_file_path, index=False)

# 限制最大插值gap
def interpolate_with_limit(df, max_gap=12):
    df_interp = df.interpolate(method='linear', limit=max_gap, limit_direction='both')
    def mask_long_gaps(series, limit=max_gap):
        is_na = series.isna()
        grp = (is_na != is_na.shift()).cumsum()
        group_sizes = is_na.groupby(grp).transform('sum')
        over_limit = (is_na) & (group_sizes > limit)
        return series.where(~over_limit)
    for col in df.columns:
        df_interp[col] = mask_long_gaps(df[col]).combine_first(df_interp[col])
    return df_interp

def solar_angles(dt, phi, lambda_, lambda_std):
    phi_rad = np.deg2rad(phi)
    if isinstance(dt, pd.Timestamp):
        n = dt.day_of_year
    else:
        n = dt.timetuple().tm_yday
    Gamma = 2 * np.pi * (n - 1) / 365
    delta_rad = (0.006918 - 0.399912 * np.cos(Gamma) + 0.070257 * np.sin(Gamma)
                 - 0.006758 * np.cos(2 * Gamma) + 0.000907 * np.sin(2 * Gamma))
    delta_deg = np.rad2deg(delta_rad)
    delta_rad = np.deg2rad(delta_deg)
    EoT = 229.2 * (0.000075 + 0.001868 * np.cos(Gamma)
                   - 0.032077 * np.sin(Gamma) - 0.014615 * np.cos(2 * Gamma)
                   + 0.000907 * np.sin(2 * Gamma))
    T_local = dt.hour + dt.minute / 60 + dt.second / 3600
    T_solar = T_local + (lambda_ - lambda_std) / 15 + EoT / 60
    H_deg = 15 * (T_solar - 12)
    H_rad = np.deg2rad(H_deg)
    sin_h = np.sin(phi_rad) * np.sin(delta_rad) + np.cos(phi_rad) * np.cos(delta_rad) * np.cos(H_rad)
    h_rad = np.arcsin(sin_h)
    h_deg = np.rad2deg(h_rad)
    zenith = 90 - h_deg
    sin_H = np.sin(H_rad)
    cos_H = np.cos(H_rad)
    x = cos_H * np.sin(phi_rad) - np.tan(delta_rad) * np.cos(phi_rad)
    y = sin_H
    A_rad = np.arctan2(y, x)
    azimuth = np.mod(np.rad2deg(A_rad), 360)
    return zenith, azimuth

def add_solar_angles(df, phi, lambda_, lambda_std):
    zeniths = []
    azimuths = []
    for t in df['datetime']:
        zenith, azimuth = solar_angles(t, phi, lambda_, lambda_std)
        zeniths.append(zenith)
        azimuths.append(azimuth)
    df['Zenith'] = zeniths
    df['Azimuth'] = azimuths
    # 行内只要有NaN非时间列全置NaN
    cols_except_time = [col for col in df.columns if col != 'datetime']
    mask_nan = df[cols_except_time].isna().any(axis=1)
    df.loc[mask_nan, cols_except_time] = pd.NA
    return df

def preprocess_all(cfg):
    # 1. 合并
    merge_raw_csv(cfg['file1_path'], cfg['file2_path'], cfg['file_flod'], cfg['merged_file_path'])
    # 2. 插值
    df = pd.read_csv(cfg['merged_file_path'])
    df_interp = interpolate_with_limit(df, max_gap=cfg.get('max_gap', 60))
    os.makedirs(os.path.dirname(cfg['interpolated_path']), exist_ok=True)
    df_interp.to_csv(cfg['interpolated_path'], index=False)
    # 3. 添加太阳角度
    df2 = pd.read_csv(cfg['interpolated_path'])
    df2['datetime'] = pd.to_datetime(df2['datetime'])
    df2 = add_solar_angles(df2, cfg['phi'], cfg['lambda_'], cfg['lambda_std'])
    os.makedirs(os.path.dirname(cfg['final_path']), exist_ok=True)
    df2.to_csv(cfg['final_path'], index=False)

