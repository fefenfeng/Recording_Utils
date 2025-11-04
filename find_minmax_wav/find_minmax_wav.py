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


def main():
    parser = argparse.ArgumentParser(description='查找目录中最长和最短的wav文件')
    parser.add_argument('directory', type=Path, help='要扫描的目录路径')
    args = parser.parse_args()

    if not args.directory.exists():
        print(f"\n目录 {args.directory} 不存在\n")
        return

    print(f"\n扫描目录: {args.directory}")
    print("=" * 100)

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
