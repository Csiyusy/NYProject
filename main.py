import argparse
import yaml
from src.preprocessing import preprocess_all
from src.train import train_lstm_model
from src.predict import run_predict
import pandas as pd
import os
import sys
import shutil

def clear_folder(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 递归删除文件夹
            except Exception as e:
                print(f"无法删除 {file_path}, 原因: {e}")
    else:
        os.makedirs(folder)

def main(mode, seq_opt, file1_path, file2_path, file_flod, scaler_X_path, scaler_Y_path, out_flod, old_model_name, model_name):
    clear_folder(out_flod)
    model_dir = os.path.dirname(model_name)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    config = {
        "file1_path": file1_path, #输入
        "file2_path": file2_path, #输入
        "file_flod": file_flod, #输入
        "merged_file_path": "./ProcessResult/ningyang_hebin.csv", #中间结果输出
        "interpolated_path": "./ProcessResult/ningyang_interp.csv", #中间结果输出
        "final_path": "./ProcessResult/ningyang_data.csv", #中间结果输出
        "phi": 35.824,
        "lambda_": 116.806,
        "lambda_std": 120,
        "max_gap": 60,
    }
    preprocess_all(config)
    print("预处理流程完成！")

    # seq_out 由 seq_opt 控制
    seq_out = 24 if seq_opt == 1 else 288

    if mode == "train":
        df = pd.read_csv(config["final_path"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        # 重叠10天的区间
        overlap_start = pd.Timestamp('2025-08-10')
        overlap_end = pd.Timestamp('2025-08-20')  # 含14号当天

        # 分段
        train_df = df[df['datetime'] < overlap_start]  # 前段
        overlap_df = df[(df['datetime'] >= overlap_start) & (df['datetime'] <= overlap_end)]  # 中间10天
        test_df = df[df['datetime'] > overlap_end]  # 后段

        # 训练集=前段+重叠段
        train_all = pd.concat([train_df, overlap_df], ignore_index=True)
        # 测试集=重叠段+后段
        test_all = pd.concat([overlap_df, test_df], ignore_index=True)

        # 保存结果
        train_all.to_csv("./ProcessResult/ningyang_train.csv", index=False)
        test_all.to_csv("./ProcessResult/ningyang_test.csv", index=False)

        train_lstm_model(
            train_csv_path = './ProcessResult/ningyang_train.csv',
            old_model_name = old_model_name,
            save_model_path = model_name,
            loss_png_path = model_name.replace('.pth', '.png'),
            batch_size = 64,
            split_ratio = 0.8,
            feature_dim = 16,
            seq_length = 960 + seq_out,
            seq_out = seq_out,
            num_epochs = 50,
            learning_rate = 0.001
        )
        print("训练流程已完成。")

        run_predict(
            test_file = './ProcessResult/ningyang_test.csv',
            model_name = model_name,
            out_flod = out_flod,
            scaler_X_path = "./scaler/scaler_X_4h.pkl",
            scaler_Y_path = "./scaler/scaler_Y_4h.pkl",
            seq_length = 960,
            seq_out = seq_out,
            input_dim = 16
        )
        print("预测流程已完成。")

    if mode == "predict":
        run_predict(
            test_file='./ProcessResult/ningyang_data.csv',
            model_name=model_name,
            out_flod=out_flod,
            scaler_X_path=scaler_X_path,
            scaler_Y_path=scaler_Y_path,
            seq_length=960,
            seq_out=seq_out,
            input_dim=16
        )
        print("预测流程已完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help="YAML配置文件路径（如不填则依赖命令行默认项）")

    # 以下参数都可以 YAML 中覆盖
    parser.add_argument("--mode", "-m", type=str, default="train", choices=["train", "predict"],
                        help="运行模式（train 或 predict），默认 train")
    parser.add_argument("--seq_opt", "-s", type=int, default=2, choices=[1, 2],
                        help="预测模式（1=超短期预测，2=短期预测），默认 2")
    parser.add_argument("--file1_path", type=str, default="./data/ningyang.csv",
                        help="站点实测数据文件路径（默认 ./data/ningyang.csv）")
    parser.add_argument("--file2_path", type=str, default="./data/shortwave_fluxdown.csv",
                        help="向下短波辐射通量文件路径（默认 ./data/shortwave_fluxdown.csv）")
    parser.add_argument("--file_flod", type=str, default="./data/xinan_yubao",
                        help="预报数据文件夹路径（默认 ./data/xinan_yubao）")
    parser.add_argument("--scaler_X_path", type=str, default="./scaler/scaler_X_4h.pkl",
                        help="输入特征X的归一化参数文件路径。（默认 ./scaler/scaler_X_4h.pkl）")
    parser.add_argument("--scaler_Y_path", type=str, default="./scaler/scaler_Y_4h.pkl",
                        help="目标输出y的归一化参数文件路径。（默认 ./scaler/scaler_Y_4h.pkl）")
    parser.add_argument("--out_flod", type=str, default="./result",
                        help="输出结果目录（默认 ./result）")
    parser.add_argument("--old_model_name", type=str, default="./model/lstm_4h_old.pth",
                        help="预训练模型路径，默认: ./model/lstm_4h_old.pth")
    parser.add_argument("--model_name", type=str, default="./model/lstm_4h.pth",
                        help="训练后保存/预测加载的模型路径，默认: ./model/lstm_4h.pth")

    args, unknown = parser.parse_known_args()

    # step1: 从YAML读取配置
    config = {}
    if args.config:
        print(f"加载配置文件: {args.config}")
        if not os.path.exists(args.config):
            print(f"配置文件 {args.config} 不存在！")
            exit(1)
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None:
            config = {}
        print("读取到配置内容：", config)

    # 2. 检测哪些参数实在命令行里“明确写了”
    cli_arg_names = set()
    for i, s in enumerate(sys.argv):
        if s.startswith("--"):
            cli_arg_names.add(s.lstrip("--").split("=")[0])
        elif s.startswith("-") and not s.startswith("--") and len(s) == 2:
            # 简写如 -m
            cli_arg_names.add(parser._option_string_actions[s].dest)

    # 3. 只覆盖这些明确CLI输入的参数
    for k, v in vars(args).items():
        if k in cli_arg_names:
            config[k] = v

    # 4. 如果还有缺的参数，最后补默认值（用argparse默认）
    for k, v in vars(args).items():
        if k not in config or config[k] is None:
            config[k] = v


    # step3: 调用主函数（参数全部从config里取，保证覆盖逻辑）
    main(
        config['mode'],
        config['seq_opt'],
        config['file1_path'],
        config['file2_path'],
        config['file_flod'],
        config['scaler_X_path'],
        config['scaler_Y_path'],
        config['out_flod'],
        config['old_model_name'],
        config['model_name']
    )
