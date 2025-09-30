#!/bin/bash

# 数据处理脚本一键运行器
# 该脚本会激活虚拟环境并运行所有数据处理脚本

echo "🚀 开始执行数据处理脚本..."
echo "📁 当前目录: $(pwd)"
echo "⏰ 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "📂 脚本目录: $SCRIPT_DIR"
echo "📂 项目目录: $PROJECT_DIR"
echo ""

# 切换到项目目录
cd "$PROJECT_DIR"

# 检查虚拟环境是否存在
if [ -d "venv" ]; then
    echo "🔧 激活虚拟环境..."
    source venv/bin/activate
    echo "✅ 虚拟环境已激活"
else
    echo "⚠️  未找到虚拟环境，使用系统Python"
fi

echo ""
echo "🐍 Python版本: $(python --version 2>&1)"
echo "📦 pip版本: $(pip --version 2>&1)"
echo ""

# 检查必要的依赖
echo "🔍 检查依赖包..."
required_packages=("pandas" "pyarrow" "chardet")
missing_packages=()

for package in "${required_packages[@]}"; do
    if ! python -c "import $package" 2>/dev/null; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "❌ 缺少以下依赖包: ${missing_packages[*]}"
    echo "📦 正在安装缺少的依赖..."
    pip install "${missing_packages[@]}"
    echo ""
else
    echo "✅ 所有依赖包已安装"
fi

echo ""
echo "="*60
echo " 开始执行数据处理脚本 "
echo "="*60

# 运行Python脚本
python "$SCRIPT_DIR/run_all_data_processing.py"

# 获取退出代码
exit_code=$?

echo ""
echo "="*60
echo " 执行完成 "
echo "="*60
echo "⏰ 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

if [ $exit_code -eq 0 ]; then
    echo "🎉 所有脚本执行成功！"
    echo "📁 请检查 formatted/ 目录下的输出文件"
else
    echo "❌ 脚本执行失败，退出代码: $exit_code"
    echo "💡 请检查上面的错误信息并重新运行"
fi

# 如果在虚拟环境中，退出虚拟环境
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "🔧 已退出虚拟环境"
fi

exit $exit_code