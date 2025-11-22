# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/11/22 12:19

@Description: 麦克风低频修复主脚本
"""
import argparse
import configparser
from pathlib import Path

from tqdm import tqdm

from utils import (
    read_wav, get_avg_amp_spec, save_wav,
    compute_spectral_difference, build_gain_curve_with_transition,
    apply_spectral_compensation, scan_target_audio_files,
    save_compensation_curve, visualize_compensation_curve
)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Microphone Low Frequency Repair Tool')
    parser.add_argument('--config', default='config.cfg', help='配置文件路径(默认: config.cfg)')
    return parser.parse_args()


def load_config(config_path: str) -> configparser.ConfigParser:
    """加载配置文件"""
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    return config


def main():
    # 1. 解析命令行和配置文件
    args = parse_arguments()
    config = load_config(args.config)

    print("=" * 60)
    print("麦克风低频修复工具")
    print("=" * 60)

    # ============ 阶段一：校准阶段 ============
    print("\n=== 阶段一：计算频谱补偿曲线 ===\n")

    # 2. 读取校准音频
    print("读取校准音频...")
    damaged_path = config['calibration_audio']['damaged_mic_path']
    reference_path = config['calibration_audio']['reference_mic_path']
    target_sr = config.getint('audio_processing', 'target_sample_rate')

    sr_damaged, audio_damaged, subtype_damaged = read_wav(damaged_path, target_sr)
    sr_ref, audio_ref, _ = read_wav(reference_path, target_sr)

    if audio_damaged is None or audio_ref is None:
        print("错误: 校准音频读取失败，程序终止")
        return

    # 3. 计算平均幅度谱
    print("计算平均幅度谱...")
    start_time = config.getfloat('audio_processing', 'start_time')
    end_time = config.getfloat('audio_processing', 'end_time')
    n_fft = config.getint('audio_processing', 'n_fft')
    hop_length = config.getint('audio_processing', 'hop_length')
    window_type = config.get('audio_processing', 'window_type')

    freqs, damaged_amp_spec = get_avg_amp_spec(
        audio_damaged, sr_damaged, start_time, end_time,
        n_fft, hop_length, window_type
    )
    _, reference_amp_spec = get_avg_amp_spec(
        audio_ref, sr_ref, start_time, end_time,
        n_fft, hop_length, window_type
    )

    # 4. 计算频谱差异
    print("计算频谱差异...")
    freq_range = eval(config.get('frequency_repair', 'repair_freq_range'))
    compensation_freqs, compensation_db = compute_spectral_difference(
        freqs, damaged_amp_spec, reference_amp_spec, freq_range
    )
    print(f"频率范围: {freq_range[0]}-{freq_range[1]} Hz")
    print(f"平均补偿值: {compensation_db.mean():.2f} dB")

    # 5. 构建增益曲线(含过渡区)
    print("构建增益曲线...")
    transition_width = config.getfloat('frequency_repair', 'transition_width')
    gain_curve = build_gain_curve_with_transition(
        freqs, compensation_freqs, compensation_db,
        freq_range, transition_width
    )
    print(f"过渡区宽度: {transition_width} Hz")

    # 6. 可选：保存补偿曲线
    if config.getboolean('output', 'save_compensation_curve'):
        curve_file = config.get('output', 'compensation_curve_file')
        save_compensation_curve(
            curve_file, compensation_freqs, compensation_db,
            freq_range, transition_width
        )

    # 7. 可选：可视化补偿曲线
    if config.getboolean('output', 'save_visualization'):
        viz_file = config.get('output', 'visualization_file')
        visualize_compensation_curve(
            freqs, damaged_amp_spec, reference_amp_spec,
            compensation_freqs, compensation_db,
            freq_range, viz_file
        )

    # ============ 阶段二：批量修复 ============
    print("\n=== 阶段二：批量修复音频文件 ===\n")

    # 8. 扫描目标文件
    print("扫描目标文件...")
    data_root_dir = config.get('batch_repair', 'data_root_dir')
    target_prefix = config.get('batch_repair', 'target_prefix')
    repaired_prefix = config.get('batch_repair', 'repaired_prefix')

    target_files = scan_target_audio_files(data_root_dir, target_prefix)
    print(f"找到 {len(target_files)} 个待修复文件\n")

    if len(target_files) == 0:
        print("警告: 未找到任何匹配的文件")
        return

    # 9. 批量处理
    success_count = 0
    for filepath in tqdm(target_files, desc="修复进度", unit="file"):
        # 读取
        sr, audio, subtype = read_wav(filepath, sr_damaged)
        if audio is None:
            print(f"\n警告: 跳过文件 {filepath}")
            continue

        # 应用补偿
        audio_repaired = apply_spectral_compensation(
            audio, sr, gain_curve,
            n_fft, hop_length, window_type
        )

        # 构造新文件名(替换前缀)
        filename = Path(filepath).name
        new_filename = filename.replace(target_prefix, repaired_prefix, 1)
        new_filepath = str(Path(filepath).parent / new_filename)

        # 保存
        save_wav(new_filepath, sr, audio_repaired, subtype)
        success_count += 1

    print(f"\n=== 修复完成 ===")
    print(f"成功修复: {success_count}/{len(target_files)} 个文件")


if __name__ == '__main__':
    main()
