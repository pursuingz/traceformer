#!/usr/bin/env python
"""
简化版脚本：读取H5文件内的字段并输出为文本或表格
（不依赖tabulate库）
"""

import h5py
import numpy as np
import os
from pathlib import Path
import argparse
import json

def explore_h5_file(file_path):
    """
    读取H5文件并返回其字段信息
    
    Args:
        file_path: H5文件路径
        
    Returns:
        list: 包含字段信息的列表
    """
    fields_info = []
    
    with h5py.File(file_path, 'r') as f:
        def traverse_group(group, prefix=''):
            """递归遍历H5文件的所有字段"""
            for key in group.keys():
                item = group[key]
                full_key = f"{prefix}/{key}" if prefix else key
                
                if isinstance(item, h5py.Dataset):
                    # 这是一个数据集
                    shape = item.shape
                    dtype = item.dtype
                    size = np.prod(shape) if shape else 0
                    
                    # 获取数据样本
                    try:
                        if len(item.shape) == 0:
                            value = item[()]
                        elif np.prod(item.shape) > 10:
                            value = f"Array(first 3): {item[...].flat[:3]}"
                        else:
                            value = item[...]
                    except:
                        value = "Unable to read"
                    
                    fields_info.append({
                        'field': full_key,
                        'shape': str(shape),
                        'dtype': str(dtype),
                        'size': size,
                        'value': str(value)[:80]
                    })
                
                elif isinstance(item, h5py.Group):
                    # 这是一个组，递归遍历
                    traverse_group(item, full_key)
        
        traverse_group(f)
    
    return fields_info

def print_simple_table(fields_info):
    """以简单表格格式打印"""
    print("\n" + "="*120)
    print(f"{'Field':<30} | {'Shape':<15} | {'Dtype':<15} | {'Size':<10} | {'Sample Value':<40}")
    print("-"*120)
    
    for info in fields_info:
        field = info['field'][:29]
        shape = info['shape'][:14]
        dtype = info['dtype'][:14]
        size = str(info['size'])[:9]
        value = info['value'][:39]
        
        print(f"{field:<30} | {shape:<15} | {dtype:<15} | {size:<10} | {value:<40}")
    
    print("="*120 + "\n")

def print_detailed_text(file_path, fields_info):
    """以详细文本格式打印"""
    print("\n" + "="*80)
    print(f"H5 File: {file_path}")
    print(f"Total Fields: {len(fields_info)}")
    print("="*80)
    
    for idx, info in enumerate(fields_info, 1):
        print(f"\n[{idx}] {info['field']}")
        print(f"     Shape:  {info['shape']}")
        print(f"     Dtype:  {info['dtype']}")
        print(f"     Size:   {info['size']} elements")
        print(f"     Value:  {info['value']}")
    
    print("\n" + "="*80 + "\n")

def save_json_format(file_path, fields_info, output_file):
    """以JSON格式保存"""
    with open(output_file, 'w') as f:
        json.dump({
            'file': file_path,
            'total_fields': len(fields_info),
            'fields': fields_info
        }, f, indent=2)
    print(f"✓ JSON已保存到: {output_file}")

def save_csv_format(file_path, fields_info, output_file):
    """以CSV格式保存"""
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['field', 'shape', 'dtype', 'size', 'value'])
        writer.writeheader()
        writer.writerows(fields_info)
    print(f"✓ CSV已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='读取H5文件内的字段信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 查看单个H5文件（表格格式）
  python explore_h5_simple.py sample/03915_001.h5
  
  # 查看单个H5文件（详细文本格式）
  python explore_h5_simple.py sample/03915_001.h5 --format detailed
  
  # 查看目录中的所有H5文件
  python explore_h5_simple.py sample/ --all
  
  # 保存为JSON格式
  python explore_h5_simple.py sample/03915_001.h5 --json output.json
  
  # 保存为CSV格式
  python explore_h5_simple.py sample/03915_001.h5 --csv output.csv
        """)
    
    parser.add_argument('path', help='H5文件路径或包含H5文件的目录')
    parser.add_argument('--format', choices=['table', 'detailed'], default='table',
                        help='输出格式（默认: table）')
    parser.add_argument('--all', action='store_true',
                        help='如果输入是目录，处理所有H5文件')
    parser.add_argument('--json', type=str, metavar='FILE',
                        help='保存为JSON格式')
    parser.add_argument('--csv', type=str, metavar='FILE',
                        help='保存为CSV格式')
    
    args = parser.parse_args()
    
    path = args.path
    
    # 处理文件或目录
    if os.path.isfile(path):
        # 单个文件
        if not path.endswith('.h5'):
            print(f"❌ 错误: {path} 不是H5文件")
            return
        
        print(f"📂 读取文件: {path}")
        try:
            fields_info = explore_h5_file(path)
            
            # 输出
            if args.format == 'table':
                print_simple_table(fields_info)
            else:
                print_detailed_text(path, fields_info)
            
            # 保存
            if args.json:
                save_json_format(path, fields_info, args.json)
            if args.csv:
                save_csv_format(path, fields_info, args.csv)
        
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    elif os.path.isdir(path):
        # 目录
        h5_files = sorted([f for f in os.listdir(path) if f.endswith('.h5')])
        
        if not h5_files:
            print(f"❌ 目录中没有H5文件: {path}")
            return
        
        print(f"📂 目录: {path}")
        print(f"📊 找到 {len(h5_files)} 个H5文件\n")
        
        if args.all:
            # 处理所有H5文件
            for idx, h5_file in enumerate(h5_files, 1):
                file_path = os.path.join(path, h5_file)
                print(f"\n[{idx}/{len(h5_files)}] {h5_file}")
                print("-" * 80)
                
                try:
                    fields_info = explore_h5_file(file_path)
                    if args.format == 'table':
                        print_simple_table(fields_info)
                    else:
                        print_detailed_text(file_path, fields_info)
                except Exception as e:
                    print(f"❌ 错误: {e}\n")
        else:
            # 只显示第一个文件
            file_path = os.path.join(path, h5_files[0])
            print(f"📄 显示第一个文件: {h5_files[0]}\n")
            print(f"💡 提示: 使用 --all 标志查看所有 {len(h5_files)} 个文件\n")
            
            try:
                fields_info = explore_h5_file(file_path)
                if args.format == 'table':
                    print_simple_table(fields_info)
                else:
                    print_detailed_text(file_path, fields_info)
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    else:
        print(f"❌ 错误: {path} 不存在")

if __name__ == '__main__':
    main()
