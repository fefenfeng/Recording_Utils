# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/10/20 下午1:45

@Description: 按记录的excel文档，整理AU录制的音频文件到三层目录结构
"""
import argparse
import configparser
from pathlib import Path

from utils import build_file_index, parse_mapping_xlsx, validate_mapping, organize_files


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


def main():
    parser = argparse.ArgumentParser(description='整理AU录制的音频文件')
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
    organize_files(index, mapping, config['prefixes'], config['root_dir'])

    print("🎉 文件整理完成！")


if __name__ == "__main__":
    main()
