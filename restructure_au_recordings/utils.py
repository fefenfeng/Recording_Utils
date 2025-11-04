# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/10/20 下午1:45

@Description: restructure_au_recordings -- 功能性模块化工具函数
"""
import shutil
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, List

import pandas as pd
from tqdm import tqdm


def build_file_index(data_dir: Path, prefixes: List[str]) -> Dict[Tuple[str, str], Path]:
    """
    遍历并为data_dir下的wav文件创建(前,后缀)-路径索引字典
    :param data_dir: au多轨录音会话录音文件所在目录路径
    :param prefixes: au多轨录音会话通道前缀列表
    :return: {(prefix, suffix_str): wav_path} (前,后缀)-路径索引字典
    """
    index: Dict[Tuple[str, str], Path] = {}

    for wav in data_dir.glob("*.wav"):
        stem = wav.stem
        for prefix in prefixes:
            prefix_tag = f"{prefix}_"
            if stem.lower().startswith(prefix_tag.lower()):
                suffix = stem[len(prefix_tag):]
                index[(prefix, suffix)] = wav
                break
    return index


def parse_mapping_xlsx(mapping_file: Path) -> List[Tuple[str, List[str]]]:
    """
    解析记录excel文件(participant|session_id - 后缀)；文件格式示例：

    participant | 1st | 2nd | 3rd | ...

    327811      | 004 | 001_5 | ...

    009067      | 0664 | ...

    :param mapping_file: (participant|session_id - 后缀) excel记录文件路径
    :return: [(person_str, [suffix1, suffix2, ...]), ...]，participant与后缀列表映射关系
    """
    df = pd.read_excel(mapping_file, dtype=str)
    mapping: List[Tuple[str, List[str]]] = []

    for _, row in df.iterrows():
        person = str(row.iloc[0]).strip()
        if not person:
            continue    # 跳过空行

        suffixes: List[str] = []
        for val in row.iloc[1:]:
            if pd.isna(val):
                continue
            suffix_str = str(val).strip()
            if suffix_str:
                suffixes.append(suffix_str)
        mapping.append((person, suffixes))
    return mapping


def validate_mapping(mapping: List[Tuple[str, List[str]]]) -> None:
    """
    验证 mapping 的合理性，打印没有后缀编号的 participant 以及重复的 participant
    :param mapping: participant 与后缀列表映射关系
    :return: None
    """
    empty_people = [p for p, suffixes in mapping if not suffixes]
    print(f"没有任何后缀编号的 participant: {empty_people}")
    print(f"唯一 participant 数: {len(set(p for p, _ in mapping))}")

    p_counts = Counter(p for p, _ in mapping)
    duplicates = [p for p, count in p_counts.items() if count > 1]
    print(f"重复的 participant: {duplicates}")


def organize_files(
    index: Dict[Tuple[str, str], Path],
    mapping: List[Tuple[str, List[str]]],
    prefixes: List[str],
    root_dir: Path
) -> None:
    """
    根据 mapping 复制到 root/实验人/session_id/
    :param index: 由build_file_index创建的(前,后缀)-路径索引字典
    :param mapping: 由parse_mapping_xlsx创建的participant与后缀列表映射关系
    :param prefixes: 通道前缀列表
    :param root_dir: 目标根目录路径
    :return: None
    """
    total_tasks = sum(len(suffixes) for _, suffixes in mapping) * len(prefixes)
    pbar = tqdm(total=total_tasks, unit="file", desc="Copying wav files")

    for person, suffix_list in mapping:
        person_dir = person
        while (root_dir / person_dir).exists():
            print(person_dir)
            person_dir += '0'

        for session_id, suffix in enumerate(suffix_list, start=1):
            dest_dir = root_dir / person_dir / str(session_id)
            dest_dir.mkdir(parents=True, exist_ok=True)

            for prefix in prefixes:
                key = (prefix, suffix)
                if key not in index:
                    print(f"[缺失] {prefix}_{suffix}.wav")
                    pbar.update(1)
                    continue

                src = index[key]
                dst = dest_dir / src.name
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[错误] 复制 {src} -> {dst} 失败: {e}")
                pbar.update(1)

    pbar.close()
