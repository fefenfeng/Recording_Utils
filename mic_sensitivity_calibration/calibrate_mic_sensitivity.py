# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng
@Date: 2025/10/20 下午3:58
@Description: 根据麦克风灵敏度结果对音频文件进行校准
"""
import os
import argparse
import configparser
import shutil
from collections import defaultdict

import numpy as np

from utils import (read_wav, save_wav, load_sensitivity_results, calculate_gain_factor,
                   parse_channel_mapping, scan_all_audio_files, apply_gain_to_audio)


def load_config(cfg_path: str) -> dict:
    """
    加载配置文件
    :param cfg_path: 配置文件路径
    :return: 配置字典
    """
    config = configparser.ConfigParser()
    config.read(cfg_path, encoding='utf-8')

    cfg_dict = {
        'sensitivity_result_json': config.get('calibration_paths', 'sensitivity_result_json'),
        'source_dir': config.get('calibration_paths', 'source_dir'),
        'target_dir': config.get('calibration_paths', 'target_dir'),
        'channel_mapping': parse_channel_mapping(config.get('calibration_settings', 'channel_mapping')),
        'reference_target_prefix': config.get('calibration_settings', 'reference_target_prefix'),
        'freq_ranges_for_calibration': eval(config.get('calibration_settings', 'freq_ranges_for_calibration'))
    }

    return cfg_dict


def extract_global_level(results_data: dict) -> dict:
    """
    从完整结果中提取global_level部分
    :param results_data: 完整的灵敏度结果
    :return: global_level数据
    """
    if 'results' in results_data and 'global_level' in results_data['results']:
        return results_data['results']['global_level']
    return {}


def compute_gain_factors(global_level: dict, channel_mapping: dict,
                         freq_ranges: list[tuple], reference_target_prefix: str) -> dict:
    """
    计算各通道的增益因子
    :param global_level: global_level数据
    :param channel_mapping: 通道映射 {result_prefix: target_prefix}
    :param freq_ranges: 用于校准的频率范围列表
    :param reference_target_prefix: 参考通道的目标前缀
    :return: {target_prefix: gain_factor}
    """
    gain_factors = {}
    reverse_mapping = {v: k for k, v in channel_mapping.items()}

    for target_prefix, result_prefix in reverse_mapping.items():
        # 参考通道增益为1.0
        if target_prefix == reference_target_prefix:
            gain_factors[target_prefix] = 1.0
            continue

        # 检查是否在global_level中
        if result_prefix not in global_level:
            print(f"Warning: {result_prefix} not found in global_level, setting gain to 1.0")
            gain_factors[target_prefix] = 1.0
            continue

        # 提取指定频率范围的dB值
        db_values = []
        for freq_range in freq_ranges:
            freq_range_str = str(freq_range)

            if freq_range_str in global_level[result_prefix]:
                db_values.append(global_level[result_prefix][freq_range_str])
            else:
                print(f"Warning: Frequency range {freq_range} not found for {result_prefix}")

        # 计算增益因子
        if db_values:
            gain_factors[target_prefix] = calculate_gain_factor(db_values)
        else:
            gain_factors[target_prefix] = 1.0

    return gain_factors


def classify_files(audio_files: list[str], gain_factors: dict, reference_target_prefix: str) -> dict:
    """
    对文件进行分类
    :param audio_files: 所有音频文件相对路径列表
    :param gain_factors: 增益因子字典
    :param reference_target_prefix: 参考通道前缀
    :return: {
        'reference': [files],
        'calibrate': {prefix: [files]},
        'others': [files]
    }
    """
    classification = {
        'reference': [],
        'calibrate': defaultdict(list),
        'others': []
    }

    for filepath in audio_files:
        filename = os.path.basename(filepath)
        matched = False

        # 检查是否匹配任何需要处理的前缀
        for target_prefix in gain_factors.keys():
            if filename.startswith(target_prefix):
                if target_prefix == reference_target_prefix:
                    classification['reference'].append(filepath)
                else:
                    classification['calibrate'][target_prefix].append(filepath)
                matched = True
                break

        if not matched:
            classification['others'].append(filepath)

    return classification


def process_files(cfg: dict, gain_factors: dict, file_classification: dict,
                  global_level: dict, reverse_mapping: dict) -> dict:
    """
    处理所有文件(复制或校准)
    :param cfg: 配置字典
    :param gain_factors: 增益因子字典
    :param file_classification: 文件分类结果
    :param global_level: global_level数据用于显示信息
    :param reverse_mapping: {target_prefix: result_prefix}
    :return: 处理统计信息字典
    """
    source_dir = cfg['source_dir']
    target_dir = cfg['target_dir']

    stats = {
        'reference_copied': 0,
        'calibrated': defaultdict(int),
        'others_copied': 0
    }

    print("\nProcessing files...")

    # 1. 处理参考通道文件(直接复制)
    print(f"\nCopying reference files ({cfg['reference_target_prefix']})...")
    for rel_path in file_classification['reference']:
        src_path = os.path.join(source_dir, rel_path)
        dst_path = os.path.join(target_dir, rel_path)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        stats['reference_copied'] += 1

    # 2. 处理需要校准的文件
    print("\nCalibrating files...")
    for target_prefix, file_list in file_classification['calibrate'].items():
        gain = gain_factors[target_prefix]
        result_prefix = reverse_mapping.get(target_prefix, "Unknown")

        # 显示该通道的详细校准信息
        print(f"  Processing {target_prefix} (from {result_prefix}, gain={gain:.4f})...")

        # 显示该通道在各频率范围的原始dB差异
        if result_prefix in global_level:
            db_info = ", ".join([f"{fr}: {db:.2f}dB"
                                 for fr, db in sorted(global_level[result_prefix].items())])
            print(f"    Original differences: {db_info}")

        for rel_path in file_list:
            src_path = os.path.join(source_dir, rel_path)
            dst_path = os.path.join(target_dir, rel_path)

            # 读取音频
            sr, data = read_wav(src_path, target_rate=48000)
            if data is None:
                print(f"    Warning: Failed to read {rel_path}")
                continue

            # 应用增益
            calibrated_data = apply_gain_to_audio(data, gain)

            # 保存
            save_wav(dst_path, sr, calibrated_data)
            stats['calibrated'][target_prefix] += 1

    # 3. 处理其他文件(直接复制)
    if file_classification['others']:
        print("\nCopying other files...")
        for rel_path in file_classification['others']:
            src_path = os.path.join(source_dir, rel_path)
            dst_path = os.path.join(target_dir, rel_path)

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            stats['others_copied'] += 1

    return stats


def print_summary(cfg: dict, gain_factors: dict, stats: dict,
                  global_level: dict, reverse_mapping: dict):
    """打印处理摘要"""
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)

    print(f"\nSource Directory: {cfg['source_dir']}")
    print(f"Target Directory: {cfg['target_dir']}")
    print(f"Sensitivity Results: {cfg['sensitivity_result_json']}")

    print("\n--- Gain Factors ---")
    for target_prefix, gain in sorted(gain_factors.items()):
        if target_prefix == cfg['reference_target_prefix']:
            print(f"  {target_prefix}: 1.0000 (reference)")
        else:
            result_prefix = reverse_mapping.get(target_prefix, "Unknown")

            # 显示映射关系和增益因子
            print(f"  {target_prefix} ← {result_prefix}: gain={gain:.4f}")

            # 显示原始dB差异和平均值
            if result_prefix in global_level:
                db_values = []
                for freq_range in cfg['freq_ranges_for_calibration']:
                    freq_range_str = str(freq_range)
                    if freq_range_str in global_level[result_prefix]:
                        db = global_level[result_prefix][freq_range_str]
                        db_values.append(db)
                        print(f"      {freq_range_str}: {db:.2f} dB")

                if db_values:
                    avg_db = np.mean(db_values)
                    print(f"      Average: {avg_db:.2f} dB")

    print("\n--- Files Processed ---")
    print(f"  Reference ({cfg['reference_target_prefix']}): {stats['reference_copied']} files copied")

    for target_prefix, count in sorted(stats['calibrated'].items()):
        result_prefix = reverse_mapping.get(target_prefix, "Unknown")
        print(f"  Calibrated ({target_prefix} ← {result_prefix}): {count} files processed")

    if stats['others_copied'] > 0:
        print(f"  Others: {stats['others_copied']} files copied")

    total = stats['reference_copied'] + sum(stats['calibrated'].values()) + stats['others_copied']
    print(f"\nTotal: {total} files processed successfully")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Calibrate microphone sensitivity')
    parser.add_argument('--config', default='config.cfg', help='配置文件路径(默认: config.cfg)')
    args = parser.parse_args()

    # 验证配置文件是否存在
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print(f"Current working directory: {os.getcwd()}")
        return

    # 加载配置
    print(f"Loading configuration from: {args.config}")
    cfg = load_config(args.config)

    # 加载灵敏度结果
    print(f"Loading sensitivity results from: {cfg['sensitivity_result_json']}")
    results_data = load_sensitivity_results(cfg['sensitivity_result_json'])

    if not results_data:
        print("Error: Failed to load sensitivity results!")
        return

    global_level = extract_global_level(results_data)

    if not global_level:
        print("Error: No global_level data found in results!")
        return

    # 计算增益因子
    print("\nComputing gain factors...")
    gain_factors = compute_gain_factors(
        global_level,
        cfg['channel_mapping'],
        cfg['freq_ranges_for_calibration'],
        cfg['reference_target_prefix']
    )

    # 扫描源目录
    print(f"\nScanning source directory: {cfg['source_dir']}")
    audio_files = scan_all_audio_files(cfg['source_dir'])

    if not audio_files:
        print("Error: No audio files found in source directory!")
        return

    print(f"Found {len(audio_files)} audio files")

    # 分类文件
    reverse_mapping = {v: k for k, v in cfg['channel_mapping'].items()}
    file_classification = classify_files(
        audio_files,
        gain_factors,
        cfg['reference_target_prefix']
    )

    # 处理文件
    stats = process_files(cfg, gain_factors, file_classification, global_level, reverse_mapping)

    # 打印摘要
    print_summary(cfg, gain_factors, stats, global_level, reverse_mapping)


if __name__ == '__main__':
    main()
