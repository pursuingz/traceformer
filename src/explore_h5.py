#!/usr/bin/env python
"""
脚本：读取H5文件内的字段并输出为文本或表格
用途：探索H5文件的结构和数据
"""

import h5py
import numpy as np
import os
from pathlib import Path
import argparse
from tabulate import tabulate
import json

def explore_h5_file(file_path):
    """
    读取H5文件并返回其字段信息
    
    Args:
        file_path: H5文件路径
        
    Returns:
        dict: 包含字段信息的字典
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
                        'Field': full_key,
                        'Shape': str(shape),
                        'Dtype': str(dtype),
                        'Size': size,
                        'Sample Value': str(value)[:100]  # 限制长度
                    })
                
                elif isinstance(item, h5py.Group):
                    # 这是一个组，递归遍历
                    traverse_group(item, full_key)
        
        traverse_group(f)
    
    return fields_info

def print_table_format(fields_info):
    """以表格格式打印字段信息"""
    headers = ['Field', 'Shape', 'Dtype', 'Size', 'Sample Value']
    print("\n" + "="*150)
    print(tabulate(fields_info, headers=headers, tablefmt='grid'))
    print("="*150 + "\n")

def print_text_format(file_path, fields_info):
    """以文本格式打印字段信息"""
    print("\n" + "="*80)
    print(f"H5 File: {file_path}")
    print("="*80)
    
    for info in fields_info:
        print(f"\nField: {info['Field']}")
        print(f"  Shape: {info['Shape']}")
        print(f"  Data Type: {info['Dtype']}")
        print(f"  Total Size: {info['Size']}")
        print(f"  Sample Value: {info['Sample Value']}")
    
    print("\n" + "="*80 + "\n")

def save_json_format(file_path, fields_info, output_file):
    """以JSON格式保存字段信息"""
    with open(output_file, 'w') as f:
        json.dump(fields_info, f, indent=2)
    print(f"✓ JSON信息已保存到: {output_file}")

def save_text_format(file_path, fields_info, output_file):
    """以文本格式保存字段信息"""
    with open(output_file, 'w') as f:
        f.write(f"H5 File: {file_path}\n")
        f.write("="*80 + "\n\n")
        
        for info in fields_info:
            f.write(f"Field: {info['Field']}\n")
            f.write(f"  Shape: {info['Shape']}\n")
            f.write(f"  Data Type: {info['Dtype']}\n")
            f.write(f"  Total Size: {info['Size']}\n")
            f.write(f"  Sample Value: {info['Sample Value']}\n\n")
    
    print(f"✓ 文本信息已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='读取H5文件内的字段并输出为表格或文本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看单个H5文件（表格格式）
  python explore_h5.py sample/03915_001.h5
  
  # 查看单个H5文件（文本格式）
  python explore_h5.py sample/03915_001.h5 --format text
  
  # 查看目录中的所有H5文件
  python explore_h5.py sample/ --all
  
  # 保存为JSON格式
  python explore_h5.py sample/03915_001.h5 --save-json output.json
  
  # 保存为文本格式
  python explore_h5.py sample/03915_001.h5 --save-text output.txt
        """)
    
    parser.add_argument('path', help='H5文件路径或包含H5文件的目录')
    parser.add_argument('--format', choices=['table', 'text'], default='table',
                        help='输出格式（默认: table）')
    parser.add_argument('--all', action='store_true',
                        help='如果输入是目录，则处理所有H5文件')
    parser.add_argument('--save-json', type=str,
                        help='保存为JSON格式到指定文件')
    parser.add_argument('--save-text', type=str,
                        help='保存为文本格式到指定文件')
    
    args = parser.parse_args()
    
    path = args.path
    
    # 处理文件或目录
    if os.path.isfile(path):
        # 单个文件
        if not path.endswith('.h5'):
            print(f"❌ 错误: {path} 不是H5文件")
            return
        
        print(f"📂 读取文件: {path}")
        fields_info = explore_h5_file(path)
        
        # 输出
        if args.format == 'table':
            print_table_format(fields_info)
        else:
            print_text_format(path, fields_info)
        
        # 保存
        if args.save_json:
            save_json_format(path, fields_info, args.save_json)
        if args.save_text:
            save_text_format(path, fields_info, args.save_text)
    
    elif os.path.isdir(path):
        # 目录
        h5_files = sorted([f for f in os.listdir(path) if f.endswith('.h5')])
        
        if not h5_files:
            print(f"❌ 目录中没有H5文件: {path}")
            return
        
        if args.all:
            # 处理所有H5文件
            print(f"📂 目录: {path}")
            print(f"📊 找到 {len(h5_files)} 个H5文件\n")
            
            for h5_file in h5_files:
                file_path = os.path.join(path, h5_file)
                print(f"\n{'='*80}")
                print(f"文件: {h5_file}")
                print('='*80)
                
                try:
                    fields_info = explore_h5_file(file_path)
                    
                    if args.format == 'table':
                        print_table_format(fields_info)
                    else:
                        print_text_format(file_path, fields_info)
                except Exception as e:
                    print(f"❌ 处理文件失败: {e}")
        else:
            # 只显示第一个文件
            file_path = os.path.join(path, h5_files[0])
            print(f"📂 目录: {path}")
            print(f"📊 找到 {len(h5_files)} 个H5文件")
            print(f"📄 显示第一个文件: {h5_files[0]}\n")
            print(f"提示: 使用 --all 标志查看所有文件\n")
            
            fields_info = explore_h5_file(file_path)
            
            if args.format == 'table':
                print_table_format(fields_info)
            else:
                print_text_format(file_path, fields_info)
    
    else:
        print(f"❌ 错误: {path} 不存在")

if __name__ == '__main__':
    main()
