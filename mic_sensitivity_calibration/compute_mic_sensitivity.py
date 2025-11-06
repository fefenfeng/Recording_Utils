# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/10/20 下午3:58

@Description: 计算麦克风灵敏度差异
"""
import os
import argparse
import configparser
import json
from collections import defaultdict

import numpy as np

from utils import (read_wav, get_avg_amp_spec, scan_audio_files,
                   compute_avg_amplitude_in_freq_range, compute_sensitivity_diff)


def load_config(cfg_path: str) -> dict:
    """
    加载配置文件
    :param cfg_path: 配置文件路径
    :return: 配置字典
    """
    config = configparser.ConfigParser()
    config.read(cfg_path, encoding='utf-8')

    cfg_dict = {
        'root_dir': config.get('paths', 'root_dir'),
        'output_dir': config.get('paths', 'output_dir'),
        'mic_prefixes': [x.strip() for x in config.get('channels', 'mic_prefixes').split(',')],
        'reference_mic': config.get('channels', 'reference_mic'),
        'target_sample_rate': config.getint('audio_processing', 'target_sample_rate'),
        'start_time': config.getfloat('audio_processing', 'start_time'),
        'end_time': config.getfloat('audio_processing', 'end_time'),
        'n_fft': config.getint('audio_processing', 'n_fft'),
        'hop_length': config.getint('audio_processing', 'hop_length'),
        'window_type': config.get('audio_processing', 'window_type'),
        'freq_ranges': eval(config.get('frequency_analysis', 'freq_ranges')),
        'save_to_file': config.getboolean('output', 'save_to_file'),
        'output_filename': config.get('output', 'output_filename')
    }

    return cfg_dict


def compute_amplitudes(file_structure: dict, cfg: dict) -> dict:
    """
    计算所有文件在各频率范围的平均幅度
    :param file_structure: 文件组织结构
    :param cfg: 配置字典
    :return: {distance: {session: {mic: {freq_range: amplitude}}}}
    """
    amplitude_results = {}

    for distance, sessions in file_structure.items():
        amplitude_results[distance] = {}

        for session, mics in sessions.items():
            amplitude_results[distance][session] = {}

            for mic_prefix, filepath in mics.items():
                # 读取音频
                sr, data, _ = read_wav(filepath, cfg['target_sample_rate'])
                if data is None:
                    continue

                # 计算平均幅度谱
                freqs, amp_spec = get_avg_amp_spec(
                    data, sr, cfg['start_time'], cfg['end_time'],
                    cfg['n_fft'], cfg['hop_length'], cfg['window_type']
                )

                # 计算各频率范围的平均幅度
                amplitude_results[distance][session][mic_prefix] = {}
                for freq_range in cfg['freq_ranges']:
                    avg_amp = compute_avg_amplitude_in_freq_range(freqs, amp_spec, freq_range)
                    amplitude_results[distance][session][mic_prefix][freq_range] = avg_amp

    return amplitude_results


def compute_differences(amplitude_results: dict, reference_mic: str, mic_prefixes: list[str]) -> dict:
    """
    计算灵敏度差异(session级、distance级、global级)
    :param amplitude_results: 幅度计算结果
    :param reference_mic: 参考麦克风
    :param mic_prefixes: 所有麦克风前缀
    :return: 差异结果字典
    """
    # 获取非参考麦克风列表
    target_mics = [mic for mic in mic_prefixes if mic != reference_mic]

    results = {
        'session_level': {},
        'distance_level': {},
        'global_level': {}
    }

    # 用于distance和global级别的累积
    distance_diffs = defaultdict(lambda: defaultdict(list))
    global_diffs = defaultdict(list)

    # Session级别
    for distance, sessions in amplitude_results.items():
        for session, mics in sessions.items():
            if reference_mic not in mics:
                print(f"Warning: Reference mic {reference_mic} not found in {distance}/{session}")
                continue

            session_key = f"{distance}/{session}"
            results['session_level'][session_key] = {}

            for target_mic in target_mics:
                if target_mic not in mics:
                    print(f"Warning: Target mic {target_mic} not found in {distance}/{session}")
                    continue

                results['session_level'][session_key][target_mic] = {}

                for freq_range in mics[reference_mic].keys():
                    ref_amp = mics[reference_mic][freq_range]
                    target_amp = mics[target_mic][freq_range]
                    diff = compute_sensitivity_diff(target_amp, ref_amp)

                    results['session_level'][session_key][target_mic][freq_range] = diff

                    # 为distance和global级别累积
                    distance_diffs[distance][target_mic].append((freq_range, diff))
                    global_diffs[target_mic].append((freq_range, diff))

    # Distance级别 - 对每个distance下的session求平均
    for distance, mic_data in distance_diffs.items():
        results['distance_level'][distance] = {}

        for target_mic, diff_list in mic_data.items():
            results['distance_level'][distance][target_mic] = {}

            # 按频率范围分组
            freq_range_diffs = defaultdict(list)
            for freq_range, diff in diff_list:
                freq_range_diffs[freq_range].append(diff)

            # 计算平均
            for freq_range, diffs in freq_range_diffs.items():
                avg_diff = np.mean(diffs)
                results['distance_level'][distance][target_mic][freq_range] = avg_diff

    # Global级别 - 所有distance/session的总体平均
    for target_mic, diff_list in global_diffs.items():
        results['global_level'][target_mic] = {}

        # 按频率范围分组
        freq_range_diffs = defaultdict(list)
        for freq_range, diff in diff_list:
            freq_range_diffs[freq_range].append(diff)

        # 计算平均
        for freq_range, diffs in freq_range_diffs.items():
            avg_diff = np.mean(diffs)
            results['global_level'][target_mic][freq_range] = avg_diff

    return results


def format_freq_range(freq_range: tuple) -> str:
    """
    格式化频率范围为字符串
    :param freq_range: 频率范围元组 (min_freq, max_freq)
    :return: 格式化字符串
    """
    return f"[{freq_range[0]}-{freq_range[1]}Hz]"


def print_results(results: dict, reference_mic: str):
    """
    打印结果到控制台
    :param results: 差异结果字典
    :param reference_mic: 参考麦克风
    :return: None
    """
    print("\n" + "=" * 80)
    print("MIC SENSITIVITY DIFFERENCE ANALYSIS")
    print(f"Reference Microphone: {reference_mic}")
    print("=" * 80)

    # Session Level
    print("\n=== SESSION LEVEL ===")
    for session_key, mics in sorted(results['session_level'].items()):
        print(f"\n{session_key}:")
        for mic, freq_diffs in sorted(mics.items()):
            diffs_str = ", ".join([f"{format_freq_range(fr)}: {diff:.2f} dB"
                                   for fr, diff in sorted(freq_diffs.items())])
            print(f"  {mic} vs {reference_mic}: {diffs_str}")

    # Distance Level
    print("\n=== DISTANCE LEVEL ===")
    for distance, mics in sorted(results['distance_level'].items()):
        print(f"\n{distance} (averaged across sessions):")
        for mic, freq_diffs in sorted(mics.items()):
            diffs_str = ", ".join([f"{format_freq_range(fr)}: {diff:.2f} dB"
                                   for fr, diff in sorted(freq_diffs.items())])
            print(f"  {mic} vs {reference_mic}: {diffs_str}")

    # Global Level
    print("\n=== GLOBAL LEVEL ===")
    print("\nOverall average across all distances and sessions:")
    for mic, freq_diffs in sorted(results['global_level'].items()):
        diffs_str = ", ".join([f"{format_freq_range(fr)}: {diff:.2f} dB"
                               for fr, diff in sorted(freq_diffs.items())])
        print(f"  {mic} vs {reference_mic}: {diffs_str}")

    print("\n" + "=" * 80 + "\n")


def save_results(results: dict, output_path: str, reference_mic: str):
    """
    保存结果到JSON文件
    :param results: 差异结果字典
    :param output_path: 输出文件路径
    :param reference_mic: 参考麦克风
    :return: None
    """

    # 转换tuple key和numpy类型为JSON可序列化类型
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {(str(k) if isinstance(k, tuple) else k): convert_keys(v)
                    for k, v in obj.items()}
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)  # 将numpy标量转换为Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # 将numpy数组转换为Python list
        return obj

    output_data = {
        'reference_microphone': reference_mic,
        'results': convert_keys(results)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute microphone sensitivity differences')
    parser.add_argument('--config', default='config.cfg', help='配置文件路径(默认: config.cfg)')
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    print(f"Configuration loaded from: {args.config}")
    print(f"Root directory: {cfg['root_dir']}")
    print(f"Microphones: {cfg['mic_prefixes']}")
    print(f"Reference: {cfg['reference_mic']}")
    print(f"Frequency ranges: {cfg['freq_ranges']}")

    # 扫描音频文件
    print("\nScanning audio files...")
    file_structure = scan_audio_files(cfg['root_dir'], cfg['mic_prefixes'])

    if not file_structure:
        print("Error: No audio files found!")
        return

    total_files = sum(len(mics) for dist in file_structure.values()
                      for mics in dist.values())
    print(f"Found {total_files} audio files")

    # 计算幅度
    print("\nComputing amplitudes...")
    amplitude_results = compute_amplitudes(file_structure, cfg)

    # 计算差异
    print("Computing sensitivity differences...")
    diff_results = compute_differences(amplitude_results, cfg['reference_mic'], cfg['mic_prefixes'])

    # 打印结果
    print_results(diff_results, cfg['reference_mic'])

    # 保存结果(如果配置要求)
    if cfg['save_to_file']:
        output_path = os.path.join(cfg['output_dir'], cfg['output_filename'])
        save_results(diff_results, output_path, cfg['reference_mic'])


if __name__ == '__main__':
    main()
