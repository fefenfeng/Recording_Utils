# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/11/28 17:03

@Description: AU多轨录音click信号对齐 - 起始点偏移量检测与音频对齐
"""
import os
import argparse
import configparser
import warnings
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from scipy.io.wavfile import read, write, WavFileWarning
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm


def load_config(config_path: str = "config.cfg") -> configparser.ConfigParser:
    """
    加载配置文件
    :param config_path: 配置文件路径
    :return: 配置对象
    """
    config = configparser.ConfigParser()

    # 如果未指定路径,使用脚本同目录下的config.cfg
    if not os.path.isabs(config_path):
        script_dir = Path(__file__).parent
        config_path = script_dir / config_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    config.read(config_path, encoding='utf-8')
    return config


def find_nearest_candidate(offset: int, candidates: List[int]) -> Tuple[int, int]:
    """
    找到最接近的候选偏移量
    :param offset: 实际检测到的偏移量
    :param candidates: 候选偏移量列表
    :return: 最接近的候选值和距离
    """
    if not candidates:
        return offset, 0

    candidates_array = np.array(candidates)
    distances = np.abs(candidates_array - offset)
    nearest_idx = np.argmin(distances)

    # 强制转换为int避免numpy类型解包问题
    return int(candidates_array[nearest_idx]), int(distances[nearest_idx])


def find_peak_in_window(signal: np.ndarray, sr: int, start_s: float, end_s: float) -> int:
    """
    在指定时间窗口内找到最大幅度点
    :param signal: 音频信号数组
    :param sr: 采样率
    :param start_s: 窗口起始时间(秒)
    :param end_s: 窗口结束时间(秒)
    :return: 峰值在整个信号中的绝对位置(样本点索引)
    """
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)

    if end_sample > len(signal):
        end_sample = len(signal)

    window = signal[start_sample:end_sample]
    peak_rel = np.argmax(np.abs(window))

    return start_sample + peak_rel


def apply_highpass_filter(signal: np.ndarray, sr: int, cutoff: float = 50, order: int = 5) -> np.ndarray:
    """
    应用高通滤波器
    :param signal: 音频信号数组
    :param sr: 采样率
    :param cutoff: 截止频率(Hz)
    :param order: 滤波器阶数
    :return: 滤波后的信号
    """
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    filtered = sosfiltfilt(sos, signal)
    return filtered


def align_audio(audio: np.ndarray, offset: int) -> np.ndarray:
    """
    应用偏移量对音频进行对齐
    :param audio: 原始音频数组
    :param offset: 偏移量(样本数), 正数=裁剪开头, 负数=开头补零
    :return: 对齐后的音频数组
    """
    if offset > 0:
        # 正偏移: 裁剪开头
        return audio[offset:]
    elif offset < 0:
        # 负偏移: 开头补零
        padding = np.zeros((-offset,) + audio.shape[1:], dtype=audio.dtype)
        return np.concatenate([padding, audio])
    else:
        return audio


def align_session_files(
        session_path: str, offset: int,
        data_dir: str, output_dir: str, pbar: tqdm) -> None:
    """
    对session下所有WAV文件应用偏移量并保存
    :param session_path: session目录路径
    :param offset: 归类后的偏移量
    :param data_dir: 数据根目录
    :param output_dir: 输出根目录
    :param pbar: tqdm进度条对象
    :return: None
    """
    # 计算输出路径(保持相对结构)
    rel_path = os.path.relpath(session_path, data_dir)
    out_session_path = os.path.join(output_dir, rel_path)
    os.makedirs(out_session_path, exist_ok=True)

    # 遍历所有WAV文件
    wav_files = [f for f in os.listdir(session_path) if f.lower().endswith('.wav')]

    for wav_file in wav_files:
        input_path = os.path.join(session_path, wav_file)
        output_path = os.path.join(out_session_path, wav_file)

        # 读取音频(抑制警告)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=WavFileWarning)
            sr, audio = read(input_path)

        original_dtype = audio.dtype

        # 应用对齐
        aligned_audio = align_audio(audio, offset)

        # 保存(保持原始dtype)
        write(output_path, sr, aligned_audio.astype(original_dtype))

        # 更新进度条
        pbar.update(1)


def process_session(
        session_path: str, click_peak: int,
        sr: int, ref_prefix: str,
        window_start: float, window_end: float, expected_start: float,
        data_dir: str, highpass_cutoff: float, candidate_offsets: List[int],
        pbar: tqdm) -> Optional[int]:
    """
    处理单个session的对齐偏移量检测
    :param session_path: session目录路径
    :param click_peak: click.wav内部峰值位置
    :param sr: 采样率
    :param ref_prefix: 参考通道前缀
    :param window_start: 搜索窗口起始时间(秒)
    :param window_end: 搜索窗口结束时间(秒)
    :param expected_start: 理论起始时间(秒)
    :param data_dir: 数据根目录
    :param highpass_cutoff: 高通滤波截止频率(Hz)
    :param candidate_offsets: 候选偏移量列表
    :param pbar: tqdm进度条对象
    :return: 归类后的偏移量(样本数), 若失败返回None
    """
    wav_files = [f for f in os.listdir(session_path) if f.lower().endswith(".wav")]

    # 找到参考通道文件
    ref_file = None
    for f in wav_files:
        if f.startswith(ref_prefix):
            ref_file = f
            break

    if not ref_file:
        rel_session = os.path.relpath(session_path, start=data_dir)
        pbar.write(f"[跳过] 未找到匹配前缀 '{ref_prefix}' 的文件于 {rel_session}")
        return None

    # 读取参考通道(抑制警告)
    ref_path = os.path.join(session_path, ref_file)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=WavFileWarning)
        ref_sr, ref_data = read(ref_path)

    if ref_sr != sr:
        rel_session = os.path.relpath(session_path, start=data_dir)
        pbar.write(f"[警告] 采样率不一致: click={sr}, ref={ref_sr} 于 {rel_session}/{ref_file}")
        return None

    ref_data = ref_data.astype(np.float32)

    # 应用高通滤波
    ref_data_filtered = apply_highpass_filter(ref_data, sr, cutoff=highpass_cutoff)

    # 在指定窗口内找到实际峰值(使用滤波后的信号)
    actual_peak = find_peak_in_window(ref_data_filtered, sr, window_start, window_end)

    # 计算理论峰值位置
    theoretical_peak = click_peak + int(expected_start * sr)

    # 计算偏移量
    offset_samples = actual_peak - theoretical_peak
    offset_seconds = offset_samples / sr

    # 归类到最近的候选值
    nearest_offset, distance = find_nearest_candidate(offset_samples, candidate_offsets)
    nearest_offset_seconds = nearest_offset / sr

    # 使用tqdm.write输出结果
    rel_session = os.path.relpath(session_path, start=data_dir)
    pbar.write(f"[Session] {rel_session}")
    pbar.write(f"  参考文件: {ref_file}")
    pbar.write(f"  实际峰值: {actual_peak} 样本 ({actual_peak/sr:.4f} s)")
    pbar.write(f"  理论峰值: {theoretical_peak} 样本 ({theoretical_peak/sr:.4f} s)")

    sign = "+" if offset_samples >= 0 else ""
    pbar.write(f"  检测偏移量: {sign}{offset_samples} 样本 ({sign}{offset_seconds:.4f} s)")

    if candidate_offsets:
        nearest_sign = "+" if nearest_offset >= 0 else ""
        pbar.write(f"  归类偏移量: {nearest_sign}{nearest_offset} 样本 ({nearest_sign}{nearest_offset_seconds:.4f} s)")
        pbar.write(f"  误差距离: {distance} 样本 ({distance/sr:.4f} s)")

    return nearest_offset


def count_total_wav_files(data_dir: str) -> int:
    """
    统计数据目录下所有WAV文件数量
    :param data_dir: 数据根目录
    :return: WAV文件总数
    """
    total = 0
    for participant in os.listdir(data_dir):
        participant_path = os.path.join(data_dir, participant)
        if not os.path.isdir(participant_path):
            continue

        for session in os.listdir(participant_path):
            session_path = os.path.join(participant_path, session)
            if os.path.isdir(session_path):
                wav_files = [f for f in os.listdir(session_path) if f.lower().endswith('.wav')]
                total += len(wav_files)

    return total


def main() -> None:
    """
    主函数: 解析命令行参数并执行偏移量检测与音频对齐
    :raises FileNotFoundError: 当配置文件或click音频文件不存在时
    """
    parser = argparse.ArgumentParser(description='AU多轨录音click信号对齐 - 起始点偏移量检测与音频对齐')
    parser.add_argument('--config', default='config.cfg', help='配置文件路径(默认: config.cfg)')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    click_wav_path = config.get('paths', 'click_wav_path')
    data_dir = config.get('paths', 'data_dir')
    output_dir = config.get('paths', 'output_dir')
    ref_prefix = config.get('alignment', 'ref_channel_prefix')
    window_start = config.getfloat('alignment', 'search_window_start')
    window_end = config.getfloat('alignment', 'search_window_end')
    expected_start = config.getfloat('alignment', 'expected_start_time')
    highpass_cutoff = config.getfloat('alignment', 'highpass_cutoff')

    # 解析候选偏移量列表
    candidate_offsets_str = config.get('alignment', 'candidate_offsets', fallback='')
    if candidate_offsets_str.strip():
        candidate_offsets = [int(x.strip()) for x in candidate_offsets_str.split(',')]
    else:
        candidate_offsets = []

    # 读取click.wav(抑制警告)
    if not os.path.isabs(click_wav_path):
        script_dir = Path(__file__).parent
        click_wav_path = script_dir / click_wav_path

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=WavFileWarning)
        sr, click_data = read(click_wav_path)

    click_data = click_data.astype(np.float32)

    # 获取click内部峰值位置
    click_peak = int(np.argmax(np.abs(click_data)))

    print("=" * 60)
    print("[INFO] Click信号信息")
    print(f"  文件路径: {click_wav_path}")
    print(f"  采样率: {sr} Hz")
    print(f"  内部峰值位置: {click_peak} 样本")
    print(f"  搜索窗口: [{window_start}, {window_end}] s")
    print(f"  理论起始时间: {expected_start} s")
    print(f"  高通滤波截止频率: {highpass_cutoff} Hz")
    if candidate_offsets:
        print(f"  候选偏移量: {candidate_offsets} 样本")
    print("=" * 60 + "\n")

    # 统计总文件数
    total_wav_files = count_total_wav_files(data_dir)

    # 遍历所有participant和session进行偏移量检测和对齐
    with tqdm(total=total_wav_files, desc="对齐处理进度", unit="文件", position=0, leave=True) as pbar:
        for participant in sorted(os.listdir(data_dir)):
            participant_path = os.path.join(data_dir, participant)
            if not os.path.isdir(participant_path):
                continue

            for session in sorted(os.listdir(participant_path)):
                session_path = os.path.join(participant_path, session)
                if not os.path.isdir(session_path):
                    continue

                # 步骤1: 检测偏移量(传入pbar)
                offset = process_session(
                    session_path,
                    click_peak,
                    sr,
                    ref_prefix,
                    window_start,
                    window_end,
                    expected_start,
                    data_dir,
                    highpass_cutoff,
                    candidate_offsets,
                    pbar
                )

                # 步骤2: 应用对齐
                if offset is not None:
                    align_session_files(
                        session_path,
                        offset,
                        data_dir,
                        output_dir,
                        pbar
                    )
                else:
                    # 若检测失败,跳过该session的文件数
                    wav_count = len([f for f in os.listdir(session_path) if f.lower().endswith('.wav')])
                    pbar.update(wav_count)

    print("\n" + "=" * 60)
    print("[完成] 所有session偏移量检测与音频对齐完毕")
    print(f"[输出] 对齐后的文件已保存至: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
