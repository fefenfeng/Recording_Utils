# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/10/20 下午3:31

@Description: 深度遍历目录,找出最长和最短的wav文件
"""
import os
import argparse
from pathlib import Path
from typing import Optional

import soundfile as sf


def get_wav_duration(wav_path: Path) -> Optional[float]:
    """
    获取wav文件的时长(s)
    :param wav_path: wav文件路径
    :return: 时长(秒)或None如果读取失败
    """
    try:
        info = sf.info(str(wav_path))
        return info.duration
    except Exception as e:
        print(f"读取文件 {wav_path} 时出错: {e}")
        return None


def find_minmax_wav(directory: Path) -> tuple[tuple[Optional[Path], float], tuple[Optional[Path], float]]:
    """
    深度遍历目录中查找最长和最短的wav文件
    :param directory: 目录路径
    :return: ((最短文件路径, 最短时长), (最长文件路径, 最长时长))
    """
    min_duration = float('inf')
    max_duration = 0
    min_file = None
    max_file = None

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = Path(root) / file
                duration = get_wav_duration(file_path)

                if duration is not None:
                    if duration < min_duration:
                        min_duration = duration
                        min_file = file_path

                    if duration > max_duration:
                        max_duration = duration
                        max_file = file_path

    return (min_file, min_duration), (max_file, max_duration)


def find_short_wavs(directory: Path, threshold: float) -> list[tuple[Path, float]]:
    """
    查找短于指定时长的wav文件
    :param directory: 目录路径
    :param threshold: 时长阈值(秒)
    :return: [(文件路径, 时长), ...]
    """
    short_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = Path(root) / file
                duration = get_wav_duration(file_path)

                if duration is not None and duration < threshold:
                    short_files.append((file_path, duration))

    # 按时长排序
    short_files.sort(key=lambda x: x[1])
    return short_files


def main():
    parser = argparse.ArgumentParser(description='查找目录中最长和最短的wav文件')
    parser.add_argument('directory', type=Path, help='要扫描的目录路径')
    parser.add_argument('-s', '--short', type=float, metavar='SECONDS',
                        help='列出短于指定秒数的所有音频文件')
    args = parser.parse_args()

    if not args.directory.exists():
        print(f"\n目录 {args.directory} 不存在\n")
        return

    print(f"\n扫描目录: {args.directory}")
    print("=" * 100)

    # 如果指定了 -s 参数
    if args.short is not None:
        short_files = find_short_wavs(args.directory, args.short)
        print(f"\n短于 {args.short} 秒的音频文件:")
        print(f"总数量: {len(short_files)}\n")

        if short_files:
            for file_path, duration in short_files:
                print(f"  {duration:6.2f}s - {file_path}")
        else:
            print(f"  未找到短于 {args.short} 秒的音频文件")
        print()

    (min_file, min_duration), (max_file, max_duration) = find_minmax_wav(args.directory)

    if min_file and max_file:
        print(f"\n最短音频文件:")
        print(f"  路径: {min_file}")
        print(f"  时长: {min_duration:.2f} 秒\n")

        print(f"\n最长音频文件:")
        print(f"  路径: {max_file}")
        print(f"  时长: {max_duration:.2f} 秒\n")
    else:
        print("\n未找到WAV文件\n")


if __name__ == '__main__':
    main()
