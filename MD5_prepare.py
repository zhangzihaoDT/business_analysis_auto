import pandas as pd

# 定义输入和输出文件路径
input_csv_path = '/Users/zihao_/Documents/coding/dataset/original/CM1小订.csv'
output_txt_path = '/Users/zihao_/Documents/coding/dataset/formatted/CM1小订.txt'

try:
    # 读取CSV文件，尝试使用utf-16编码
    df = pd.read_csv(input_csv_path, encoding='utf-16')
    
    # 将DataFrame写入TXT文件
    # 直接将MD5值逐行写入，不使用CSV格式
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for value in df.iloc[:, 0]:  # 获取第一列的所有值
            # 去除可能存在的制表符和空白字符
            clean_value = str(value).strip()
            f.write(f"{clean_value}\n")
    
    print(f"文件已成功从 '{input_csv_path}' 转换为 '{output_txt_path}'")
except FileNotFoundError:
    print(f"错误：文件 '{input_csv_path}' 未找到。请检查文件路径是否正确。")
except UnicodeDecodeError:
    print(f"编码错误：无法使用 'utf-16' 编码读取文件 '{input_csv_path}'。请尝试其他编码，例如 'latin1' 或 'cp936'。")
except Exception as e:
    print(f"发生错误：{e}")