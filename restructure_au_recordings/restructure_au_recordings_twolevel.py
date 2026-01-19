# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/10/20 下午1:45

@Description: 按记录的excel文档，整理AU录制的音频文件到两层目录结构
"""
import argparse
import configparser
from pathlib import Path

from utils import build_file_index, parse_mapping_xlsx, validate_mapping


def load_config(config_path: Path) -> dict:
    """
    加载配置文件
    :param config_path: 配置文件路径
    :return: 配置字典
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    data_dir = Path(config.get('paths', 'data_dir'))
    root_dir = Path(config.get('paths', 'root_dir'))
    mapping_xlsx = Path(config.get('paths', 'mapping_xlsx'))
    prefixes = [p.strip() for p in config.get('prefixes', 'values').split(',')]

    return {
        'data_dir': data_dir,
        'root_dir': root_dir,
        'mapping_xlsx': mapping_xlsx,
        'prefixes': prefixes
    }


def organize_files_flat(index, mapping, prefixes, root_dir):
    """
    根据 mapping 复制到 root/participant-session_id/
    :param index: 由build_file_index创建的(前,后缀)-路径索引字典
    :param mapping: 由parse_mapping_xlsx创建的participant与后缀列表映射关系
    :param prefixes: 通道前缀列表
    :param root_dir: 目标根目录路径
    :return: None
    """
    import shutil
    from tqdm import tqdm

    total_tasks = sum(len(suffixes) for _, suffixes in mapping) * len(prefixes)
    pbar = tqdm(total=total_tasks, unit="file", desc="Copying wav files")

    for person, suffix_list in mapping:
        for session_id, suffix in enumerate(suffix_list, start=1):
            dest_dir = root_dir / f"{person}-{session_id}"
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


def main():
    parser = argparse.ArgumentParser(description='整理AU录制的音频文件到两层结构')
    parser.add_argument('--config', default='config.cfg', help='配置文件路径(默认: config.cfg)')
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    print("========== 开始处理 ==========")
    print(f"数据目录: {config['data_dir']}")
    print(f"输出目录: {config['root_dir']}")
    print(f"映射文件: {config['mapping_xlsx']}")
    print(f"前缀列表: {config['prefixes']}")

    index = build_file_index(config['data_dir'], config['prefixes'])
    mapping = parse_mapping_xlsx(config['mapping_xlsx'])

    validate_mapping(mapping, index, config['prefixes'])
    organize_files_flat(index, mapping, config['prefixes'], config['root_dir'])

    print("🎉 文件整理完成！")


if __name__ == "__main__":
    main()
