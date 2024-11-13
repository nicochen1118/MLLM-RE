import os

# 定义函数将文件内容分成指定份数
def split_file(file_path, num_parts):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    lines_per_part = total_lines // num_parts

    # 创建目录存放分割后的文件
    output_dir = os.path.dirname(file_path)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # 将文件内容分割成num_parts部分
    for i in range(num_parts):
        start_idx = i * lines_per_part
        end_idx = start_idx + lines_per_part if i < num_parts - 1 else total_lines
        
        part_lines = lines[start_idx:end_idx]
        output_file = os.path.join(output_dir, f'{base_filename}_{i}.txt')

        with open(output_file, 'w') as out_f:
            out_f.writelines(part_lines)

        print(f'文件 {output_file} 已创建')

# 调用函数进行分割
file_path = '/home/nfs02/xingsy/code/data/MNRE/mnre_txt/mnre_test.txt'
split_file(file_path, 4)
