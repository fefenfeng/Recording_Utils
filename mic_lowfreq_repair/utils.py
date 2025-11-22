# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/11/22 12:19

@Description: mic_lowfreq_repair -- 功能性工具函数
"""
import os
import json
from typing import Tuple, List

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def read_wav(filepath: str, target_rate: int = 48000) -> tuple[int, np.ndarray, str] | tuple[None, None, None]:
    """
    读取wav单通道文件并重采样到目标采样率,同时获取原始音频格式
    :param filepath: wav文件路径
    :param target_rate: 目标采样率
    :return: (采样率, 数据, 原始subtype),遇异常返回(None, None, None)
    """
    try:
        # 获取原始文件的subtype信息
        info = sf.info(filepath)
        original_subtype = info.subtype

        # 使用librosa读取并重采样
        data, sample_rate = librosa.load(filepath, sr=target_rate, mono=True)

        return sample_rate, data, original_subtype
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None


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


def save_wav(filepath: str, sr: int, data: np.ndarray, subtype: str = 'float32'):
    """
    保存wav文件,保持与原始文件相同的格式
    :param filepath: 保存路径
    :param sr: 采样率
    :param data: 音频数据
    :param subtype: 音频格式(如'float32', 'float64', 'int16'等)
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        sf.write(filepath, data, sr, subtype=subtype)
    except Exception as e:
        print(f"Error saving {filepath}: {e}")


def compute_spectral_difference(
        freqs: np.ndarray,
        damaged_amp_spec: np.ndarray,
        reference_amp_spec: np.ndarray,
        freq_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算两个幅度谱在指定频率范围内的逐频点差异
    :param freqs: 频率数组
    :param damaged_amp_spec: 损坏麦克风的平均幅度谱(dB)
    :param reference_amp_spec: 参考麦克风的平均幅度谱(dB)
    :param freq_range: 频率范围 (min_freq, max_freq)
    :return: (补偿频率数组, 补偿值数组(dB))
    """
    min_freq, max_freq = freq_range
    mask = (freqs >= min_freq) & (freqs <= max_freq)

    compensation_freqs = freqs[mask]
    compensation_db = reference_amp_spec[mask] - damaged_amp_spec[mask]

    return compensation_freqs, compensation_db


def build_gain_curve_with_transition(
        freqs: np.ndarray,
        compensation_freqs: np.ndarray,
        compensation_db: np.ndarray,
        freq_range: Tuple[float, float],
        transition_width: float
) -> np.ndarray:
    """
    构建包含边界平滑的全频段增益曲线
    :param freqs: 全频段频率数组
    :param compensation_freqs: 补偿频率数组
    :param compensation_db: 补偿值数组(dB)
    :param freq_range: 补偿频率范围
    :param transition_width: 过渡区宽度(Hz)
    :return: 全频段增益曲线(线性增益,非dB)
    """
    f_low, f_high = freq_range
    gain_curve = np.ones(len(freqs))  # 默认增益为1

    # 插值补偿曲线到全频段
    interp_func = interp1d(compensation_freqs, compensation_db,
                           kind='linear', fill_value='extrapolate')

    for i, freq in enumerate(freqs):
        if freq < f_low - transition_width:
            # 低于过渡区下界：不补偿
            gain_db = 0.0
        elif f_low - transition_width <= freq < f_low:
            # 下过渡区：线性过渡
            alpha = (freq - (f_low - transition_width)) / transition_width
            comp_db = interp_func(f_low)
            gain_db = alpha * comp_db
        elif f_low <= freq <= f_high:
            # 完全补偿区
            gain_db = interp_func(freq)
        elif f_high < freq <= f_high + transition_width:
            # 上过渡区：线性过渡
            alpha = (f_high + transition_width - freq) / transition_width
            comp_db = interp_func(f_high)
            gain_db = alpha * comp_db
        else:
            # 高于过渡区上界：不补偿
            gain_db = 0.0

        # 转换dB到线性增益
        gain_curve[i] = 10 ** (gain_db / 20)

    return gain_curve


def apply_spectral_compensation(
        audio: np.ndarray,
        sample_rate: int,
        gain_curve: np.ndarray,
        n_fft: int,
        hop_length: int,
        window_type: str
) -> np.ndarray:
    """
    对音频信号应用频谱补偿(保留相位)
    :param audio: 输入音频信号
    :param sample_rate: 采样率
    :param gain_curve: 全频段增益曲线(线性增益)
    :param n_fft: FFT点数
    :param hop_length: hop长度
    :param window_type: 窗类型
    :return: 补偿后的音频信号
    """
    # STFT分解
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window_type)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # 应用增益曲线(逐帧应用)
    compensated_magnitude = magnitude * gain_curve[:, np.newaxis]

    # 重建STFT并逆变换
    compensated_stft = compensated_magnitude * np.exp(1j * phase)
    audio_compensated = librosa.istft(compensated_stft, hop_length=hop_length, window=window_type)

    return audio_compensated


def scan_target_audio_files(root_dir: str, target_prefix: str) -> List[str]:
    """
    递归扫描所有匹配目标前缀的wav文件
    :param root_dir: 根目录
    :param target_prefix: 目标前缀
    :return: 匹配文件的完整路径列表
    """
    matched_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav') and file.startswith(target_prefix):
                full_path = os.path.join(root, file)
                matched_files.append(full_path)

    return sorted(matched_files)


def save_compensation_curve(
        output_path: str,
        compensation_freqs: np.ndarray,
        compensation_db: np.ndarray,
        freq_range: Tuple[float, float],
        transition_width: float
) -> None:
    """
    保存补偿曲线到JSON文件
    :param output_path: 输出文件路径
    :param compensation_freqs: 补偿频率数组
    :param compensation_db: 补偿值数组(dB)
    :param freq_range: 频率范围
    :param transition_width: 过渡区宽度
    """
    data = {
        "freq_range": list(freq_range),
        "transition_width": transition_width,
        "compensation_data": {
            "frequencies": compensation_freqs.tolist(),
            "compensation_db": compensation_db.tolist()
        }
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"补偿曲线已保存到: {output_path}")
    except Exception as e:
        print(f"Error saving compensation curve: {e}")


def visualize_compensation_curve(
    freqs: np.ndarray,
    damaged_amp_spec: np.ndarray,
    reference_amp_spec: np.ndarray,
    compensation_freqs: np.ndarray,
    compensation_db: np.ndarray,
    freq_range: Tuple[float, float],
    output_path: str
) -> None:
    """
    可视化补偿前后的幅度谱对比以及补偿曲线
    :param freqs: 全频段频率数组
    :param damaged_amp_spec: 损坏麦克风幅度谱(dB)
    :param reference_amp_spec: 参考麦克风幅度谱(dB)
    :param compensation_freqs: 补偿频率数组
    :param compensation_db: 补偿值数组(dB)
    :param freq_range: 频率范围
    :param output_path: 输出图片路径
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 子图1: 原始幅度谱对比 (对数刻度)
    axes[0].semilogx(freqs, damaged_amp_spec, label='Damaged Mic', alpha=0.7)
    axes[0].semilogx(freqs, reference_amp_spec, label='Reference Mic', alpha=0.7)
    axes[0].axvline(freq_range[0], color='r', linestyle='--', alpha=0.5, label='Repair Range')
    axes[0].axvline(freq_range[1], color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Amplitude (dB)')
    axes[0].set_title('Original Amplitude Spectra Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].set_xlim([20, freqs[-1]])

    # 子图2: 补偿曲线 (线性刻度)
    axes[1].plot(compensation_freqs, compensation_db, color='green', linewidth=2)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Compensation (dB)')
    axes[1].set_title(f'Compensation Curve ({freq_range[0]}-{freq_range[1]} Hz)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([freq_range[0], freq_range[1]])

    # 子图3: 补偿后对比 (对数刻度)
    compensated_amp_spec = damaged_amp_spec.copy()
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    interp_func = interp1d(compensation_freqs, compensation_db,
                          kind='linear', fill_value='extrapolate')
    compensated_amp_spec[mask] += interp_func(freqs[mask])

    axes[2].semilogx(freqs, compensated_amp_spec, label='Compensated Damaged Mic', alpha=0.7)
    axes[2].semilogx(freqs, reference_amp_spec, label='Reference Mic', alpha=0.7)
    axes[2].axvline(freq_range[0], color='r', linestyle='--', alpha=0.5)
    axes[2].axvline(freq_range[1], color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Amplitude (dB)')
    axes[2].set_title('Compensated Amplitude Spectra Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which='both')
    axes[2].set_xlim([20, freqs[-1]])

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可视化图表已保存到: {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    finally:
        plt.close()
