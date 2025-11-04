# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng
@Date: 2025/10/20 下午3:58
@Description: 麦克风灵敏度计算及校准工具函数
"""
import os
import json

import librosa
import soundfile as sf
import numpy as np


def read_wav(filepath: str, target_rate: int = 48000) -> tuple[int, np.ndarray] | tuple[None, None]:
    """
    读取wav单通道文件并重采样到目标采样率(使用librosa库)
    :param filepath: wav文件路径
    :param target_rate: 目标采样率
    :return: 采样率和数据，遇异常为空
    """
    try:
        data, sample_rate = librosa.load(filepath, sr=target_rate, mono=True)
        return sample_rate, data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None


def get_avg_amp_spec(data: np.ndarray, sample_rate: int, start_time: float, end_time: float,
                     n_fft: int = 4096, hop_length: int = 2048, window_type: str = 'hann'
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    计算在指定时间范围内的平均(时间维度平均)幅度谱
    :param data: 读取音频后的np数组数据
    :param sample_rate: 采样率
    :param start_time: 起始时间
    :param end_time: 截至时间
    :param n_fft: 进行stft的fft点数
    :param hop_length: 间隔步长
    :param window_type: 窗类型
    :return: 频点，平均幅度谱
    """
    start_idx = int(start_time * sample_rate)
    end_idx = int(end_time * sample_rate)
    data = data[start_idx:end_idx]
    stft_result = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, window=window_type)
    amplitude_spectrum = np.abs(stft_result)
    avg_amplitude_spectrum = np.mean(amplitude_spectrum, axis=1)
    avg_amplitude_spectrum = 20 * np.log10(avg_amplitude_spectrum / (sample_rate / n_fft))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    return freqs, avg_amplitude_spectrum


def scan_audio_files(root_dir: str, mic_prefixes: list[str]) -> dict:
    """
    扫描音频文件并组织为层级结构
    :param root_dir: 数据根目录
    :param mic_prefixes: 麦克风通道前缀列表
    :return: {distance: {session: {mic_prefix: filepath}}}
    """
    file_structure = {}

    if not os.path.exists(root_dir):
        print(f"Error: Root directory {root_dir} does not exist")
        return file_structure

    # 遍历distance目录
    for distance_dir in sorted(os.listdir(root_dir)):
        distance_path = os.path.join(root_dir, distance_dir)
        if not os.path.isdir(distance_path):
            continue

        file_structure[distance_dir] = {}

        # 遍历session目录
        for session_dir in sorted(os.listdir(distance_path)):
            session_path = os.path.join(distance_path, session_dir)
            if not os.path.isdir(session_path):
                continue

            file_structure[distance_dir][session_dir] = {}

            # 遍历wav文件
            for filename in os.listdir(session_path):
                if not filename.endswith('.wav'):
                    continue

                # 检查是否匹配任何mic前缀
                for mic_prefix in mic_prefixes:
                    if filename.startswith(mic_prefix):
                        filepath = os.path.join(session_path, filename)
                        file_structure[distance_dir][session_dir][mic_prefix] = filepath
                        break

    return file_structure


def compute_avg_amplitude_in_freq_range(freqs: np.ndarray, amp_spec: np.ndarray,
                                        freq_range: tuple[float, float]) -> float:
    """
    计算指定频率范围内的平均幅度
    :param freqs: 频率数组
    :param amp_spec: 幅度谱数组
    :param freq_range: 频率范围元组 (min_freq, max_freq)
    :return: 该频率范围内的平均幅度(dB)
    """
    min_freq, max_freq = freq_range
    mask = (freqs >= min_freq) & (freqs <= max_freq)

    if not np.any(mask):
        print(f"Warning: No frequency points in range [{min_freq}, {max_freq}]")
        return 0.0

    return np.mean(amp_spec[mask])


def compute_sensitivity_diff(target_amp: float, ref_amp: float) -> float:
    """
    计算两个通道的灵敏度差异
    :param target_amp: 目标通道的幅度(dB)
    :param ref_amp: 参考通道的幅度(dB)
    :return: 差异值(dB) = target - reference
    """
    return target_amp - ref_amp


def load_sensitivity_results(json_path: str) -> dict:
    """
    读取灵敏度差异结果JSON文件
    :param json_path: JSON文件路径
    :return: 完整的结果字典
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return {}


def calculate_gain_factor(db_values: list[float]) -> float:
    """
    计算增益因子
    :param db_values: dB差异值列表
    :return: 增益因子 gain = 10^(-mean(ΔdB)/20)
    """
    if not db_values:
        return 1.0

    avg_db = np.mean(db_values)
    gain = 10 ** (-avg_db / 20)
    return gain


def parse_channel_mapping(mapping_str: str) -> dict:
    """
    解析通道映射字符串
    :param mapping_str: "Ear1_mic1:R_Ear1, Ear1_mic2:R_Ear2"
    :return: {result_prefix: target_prefix}
    """
    mapping = {}
    pairs = [pair.strip() for pair in mapping_str.split(',')]

    for pair in pairs:
        if ':' not in pair:
            continue
        result_prefix, target_prefix = pair.split(':')
        mapping[result_prefix.strip()] = target_prefix.strip()

    return mapping


def scan_all_audio_files(root_dir: str) -> list[str]:
    """
    递归扫描所有wav文件并返回相对路径
    :param root_dir: 根目录
    :return: 相对路径列表
    """
    audio_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)
                audio_files.append(rel_path)

    return sorted(audio_files)


def apply_gain_to_audio(data: np.ndarray, gain: float) -> np.ndarray:
    """
    对音频数据应用增益因子
    :param data: 音频数据
    :param gain: 增益因子
    :return: 校准后的音频数据
    """
    return data * gain


def save_wav(filepath: str, sr: int, data: np.ndarray):
    """
    保存wav文件
    :param filepath: 保存路径
    :param sr: 采样率
    :param data: 音频数据
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        sf.write(filepath, data, sr)
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
