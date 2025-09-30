# 数据处理脚本使用指南

## 概述

本目录包含了一套完整的数据处理脚本，用于将CSV数据转换为优化的Parquet格式。

## 脚本说明

### 核心处理脚本

1. **`order_data_to_parquet.py`** - 订单观察数据处理
   - 处理文件：`订单观察_data.csv`
   - 输出文件：`formatted/order_observation_data.parquet`
   - 功能：智能编码检测、数据类型优化、列分割处理

2. **`business_data_to_parquet.py`** - 业务数据处理
   - 处理文件：`业务数据记录_with表_表格.csv`
   - 输出文件：`formatted/business_daily_metrics.parquet`
   - 功能：日期转换、指标计算、滚动平均、数据类型优化

### 批量执行脚本

3. **`run_all_data_processing.py`** - Python批量执行器
   - 依次执行上述两个处理脚本
   - 提供详细的执行日志和错误处理
   - 统计执行时间和成功率

4. **`run_data_processing.sh`** - Shell一键执行器
   - 自动激活虚拟环境
   - 检查和安装依赖包
   - 执行Python批量处理脚本

## 使用方法

### 方法一：使用Shell脚本（推荐）

```bash
# 在项目根目录下执行
./scripts/run_data_processing.sh
```

### 方法二：使用Python脚本

```bash
# 确保在虚拟环境中
source venv/bin/activate

# 执行批量处理脚本
python scripts/run_all_data_processing.py
```

### 方法三：单独执行

```bash
# 单独执行订单数据处理
python scripts/order_data_to_parquet.py

# 单独执行业务数据处理
python scripts/business_data_to_parquet.py
```

## 依赖要求

- Python 3.7+
- pandas
- pyarrow
- chardet
- numpy

## 输出文件

所有处理后的Parquet文件将保存在 `formatted/` 目录下：

- `order_observation_data.parquet` - 订单观察数据
- `business_daily_metrics.parquet` - 业务指标数据

## 特性

✅ **智能编码检测** - 自动检测CSV文件编码
✅ **数据类型优化** - 减少内存使用和文件大小
✅ **错误处理** - 完善的异常处理和日志输出
✅ **进度跟踪** - 实时显示处理进度和执行时间
✅ **依赖检查** - 自动检查和安装必要的依赖包
✅ **虚拟环境支持** - 自动激活项目虚拟环境

## 故障排除

### 常见问题

1. **权限错误**
   ```bash
   chmod +x scripts/run_data_processing.sh
   ```

2. **依赖包缺失**
   ```bash
   pip install pandas pyarrow chardet
   ```

3. **文件编码问题**
   - 脚本会自动检测和处理多种编码格式
   - 支持UTF-8、GBK、GB2312等常见编码

4. **虚拟环境问题**
   ```bash
   # 创建虚拟环境
   python3 -m venv venv
   
   # 激活虚拟环境
   source venv/bin/activate
   
   # 安装依赖
   pip install pandas pyarrow chardet
   ```

## 日志输出

脚本执行时会显示详细的日志信息，包括：
- 文件读取状态
- 数据处理进度
- 数据质量报告
- 执行时间统计
- 错误信息（如有）

## 注意事项

- 确保输入的CSV文件存在于正确的路径
- `formatted/` 目录会自动创建
- 处理大文件时请确保有足够的内存空间
- 建议在虚拟环境中运行以避免依赖冲突