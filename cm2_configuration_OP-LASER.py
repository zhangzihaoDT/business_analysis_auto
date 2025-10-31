#!/usr/bin/env python3
"""
CM2激光雷达配置(OP-LASER)专项分析脚本

该脚本专门用于分析CM2车型的激光雷达配置数据，提供全面的OP-LASER配置统计分析功能。

主要功能：
1. 激光雷达配置分布分析（标准+Orin vs 高阶+Thor）
2. 配置与产品名称、产品类型的交叉分析
3. 配置与价格关系的深度分析
4. 基于时间的配置变化趋势分析
5. 配置选择的市场洞察和业务建议

专注分析：
- OP-LASER字段的详细统计
- 激光雷达配置的时间演变
- 不同产品线的配置偏好
- 配置对价格的影响分析

作者: AI Assistant
创建时间: 2025-10-31
最后更新: 2025-10-31
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime
import logging
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关库
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error, r2_score
    import shap
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"机器学习库导入失败: {e}")
    logging.warning("请安装相关依赖: pip install xgboost scikit-learn shap")
    ML_AVAILABLE = False

def setup_logging(log_level='INFO'):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_transposed_data(file_path):
    """加载转置后的CM2配置数据"""
    try:
        logging.info(f"正在加载转置数据: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        logging.info(f"数据加载成功，共 {len(df):,} 行，{len(df.columns)} 列")
        
        # 显示列名
        logging.info(f"数据列: {list(df.columns)}")
        
        # 转换日期字段
        date_columns = ['lock_time', 'invoice_time']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                logging.info(f"已转换 {col} 为日期格式")
        
        return df
        
    except Exception as e:
        logging.error(f"加载数据失败: {e}")
        raise

def analyze_field_values(df, field_name):
    """分析指定字段的值分布"""
    logging.info(f"=== 分析字段: {field_name} ===")
    
    if field_name not in df.columns:
        logging.error(f"字段 '{field_name}' 不存在于数据中")
        return None
    
    # 基本统计
    total_count = len(df)
    non_null_count = df[field_name].notna().sum()
    null_count = df[field_name].isna().sum()
    
    logging.info(f"总记录数: {total_count:,}")
    logging.info(f"非空记录数: {non_null_count:,} ({non_null_count/total_count*100:.2f}%)")
    logging.info(f"空值记录数: {null_count:,} ({null_count/total_count*100:.2f}%)")
    
    # 值分布统计
    value_counts = df[field_name].value_counts(dropna=False)
    unique_values = df[field_name].nunique(dropna=False)
    
    logging.info(f"不同值的数量: {unique_values}")
    logging.info(f"值分布:")
    
    result = {
        'field_name': field_name,
        'total_count': total_count,
        'non_null_count': non_null_count,
        'null_count': null_count,
        'unique_values': unique_values,
        'value_distribution': {}
    }
    
    for value, count in value_counts.items():
        percentage = count / total_count * 100
        logging.info(f"  {value}: {count:,} 条 ({percentage:.2f}%)")
        result['value_distribution'][str(value)] = {
            'count': count,
            'percentage': percentage
        }
    
    return result

def generate_cross_tabulation_report(df, row_field, col_field, title="交叉统计分析"):
    """生成格式化的交叉统计表报告"""
    if row_field not in df.columns or col_field not in df.columns:
        logging.warning(f"字段 '{row_field}' 或 '{col_field}' 不存在于数据中")
        return None
    
    # 创建交叉统计表
    cross_tab = pd.crosstab(df[row_field], df[col_field], margins=True)
    
    logging.info(f"\n=== {title} ===")
    logging.info(f"按{row_field}分析:")
    
    # 获取列名（排除'All'列）
    col_names = [col for col in cross_tab.columns if col != 'All']
    
    # 生成表头
    header = f"{'Value Display Name':<25}"
    for col in col_names:
        header += f" {col:>10}"
    logging.info(header)
    logging.info(f"{row_field}")
    
    # 生成数据行
    for row_name in cross_tab.index[:-1]:  # 排除'All'行
        row_str = f"{row_name:<25}"
        for col in col_names:
            if col in cross_tab.columns:
                count = cross_tab.loc[row_name, col]
                row_str += f" {count:>10}"
        logging.info(row_str)
    
    return cross_tab

def analyze_laser_configuration(df):
    """专门分析OP-LASER激光雷达配置"""
    logging.info("=== 激光雷达配置 (OP-LASER) 专项分析 ===")
    
    field_name = 'OP-LASER'
    result = analyze_field_values(df, field_name)
    
    if result is None:
        return None
    
    # 额外分析：与产品名称的关系
    if 'Product Name' in df.columns:
        cross_tab_product = generate_cross_tabulation_report(
            df, 'Product Name', field_name, 
            "激光雷达配置与产品名称关系"
        )
        
        if cross_tab_product is not None:
            if not hasattr(result, 'cross_analysis'):
                result['cross_analysis'] = {}
            result['cross_analysis']['by_product_name'] = cross_tab_product.to_dict()
    
    # 额外分析：与产品类型的关系
    if 'Product_Types' in df.columns:
        logging.info("\n--- 激光雷达配置与产品类型关系 ---")
        cross_tab = pd.crosstab(df[field_name], df['Product_Types'], margins=True)
        logging.info(f"交叉统计表:\n{cross_tab}")
        
        if not hasattr(result, 'cross_analysis'):
            result['cross_analysis'] = {}
        result['cross_analysis']['by_product_type'] = cross_tab.to_dict()
    
    # 额外分析：与价格的关系
    if '开票价格' in df.columns:
        logging.info("\n--- 激光雷达配置与价格关系 ---")
        price_by_laser = df.groupby(field_name)['开票价格'].agg(['count', 'mean', 'median', 'std']).round(2)
        logging.info(f"按激光雷达配置分组的价格统计:\n{price_by_laser}")
        
        result['price_analysis'] = price_by_laser.to_dict()
    
    return result

def analyze_time_based_laser_configuration(df, cutoff_date="2025-10-15"):
    """分析指定日期前后的OP-LASER配置差异"""
    logging.info(f"=== 时间分析：{cutoff_date} 前后的 OP-LASER 配置差异 ===")
    
    if 'lock_time' not in df.columns or 'OP-LASER' not in df.columns:
        logging.error("缺少必要的字段：lock_time 或 OP-LASER")
        return None
    
    # 转换截止日期
    try:
        cutoff_datetime = pd.to_datetime(cutoff_date)
        logging.info(f"分析截止日期: {cutoff_datetime.strftime('%Y-%m-%d')}")
    except Exception as e:
        logging.error(f"日期格式错误: {e}")
        return None
    
    # 过滤有效数据（非空的OP-LASER记录）
    valid_data = df[df['OP-LASER'].notna()].copy()
    logging.info(f"有效OP-LASER记录数: {len(valid_data):,}")
    
    # 分割数据：截止日期前后
    before_data = valid_data[valid_data['lock_time'] < cutoff_datetime]
    after_data = valid_data[valid_data['lock_time'] >= cutoff_datetime]
    
    logging.info(f"截止日期前记录数: {len(before_data):,}")
    logging.info(f"截止日期后记录数: {len(after_data):,}")
    
    result = {
        'cutoff_date': cutoff_date,
        'total_valid_records': len(valid_data),
        'before_count': len(before_data),
        'after_count': len(after_data)
    }
    
    # 分析OP-LASER配置分布差异
    logging.info("\n--- OP-LASER 配置分布对比 ---")
    
    if len(before_data) > 0:
        before_laser_dist = before_data['OP-LASER'].value_counts()
        before_laser_pct = before_data['OP-LASER'].value_counts(normalize=True) * 100
        
        logging.info(f"截止日期前 ({cutoff_date}) OP-LASER 分布:")
        for laser_type, count in before_laser_dist.items():
            pct = before_laser_pct[laser_type]
            logging.info(f"  {laser_type}: {count:,} 条 ({pct:.2f}%)")
        
        result['before_distribution'] = {
            'counts': before_laser_dist.to_dict(),
            'percentages': before_laser_pct.to_dict()
        }
    
    if len(after_data) > 0:
        after_laser_dist = after_data['OP-LASER'].value_counts()
        after_laser_pct = after_data['OP-LASER'].value_counts(normalize=True) * 100
        
        logging.info(f"\n截止日期后 ({cutoff_date}) OP-LASER 分布:")
        for laser_type, count in after_laser_dist.items():
            pct = after_laser_pct[laser_type]
            logging.info(f"  {laser_type}: {count:,} 条 ({pct:.2f}%)")
        
        result['after_distribution'] = {
            'counts': after_laser_dist.to_dict(),
            'percentages': after_laser_pct.to_dict()
        }
    
    # 计算变化趋势
    if len(before_data) > 0 and len(after_data) > 0:
        logging.info(f"\n--- 配置变化趋势分析 ---")
        
        # 计算各配置类型的变化
        changes = {}
        for laser_type in set(list(before_laser_dist.index) + list(after_laser_dist.index)):
            before_count = before_laser_dist.get(laser_type, 0)
            after_count = after_laser_dist.get(laser_type, 0)
            before_pct = before_laser_pct.get(laser_type, 0)
            after_pct = after_laser_pct.get(laser_type, 0)
            
            count_change = after_count - before_count
            pct_change = after_pct - before_pct
            
            changes[laser_type] = {
                'before_count': before_count,
                'after_count': after_count,
                'count_change': count_change,
                'before_percentage': before_pct,
                'after_percentage': after_pct,
                'percentage_change': pct_change
            }
            
            logging.info(f"{laser_type}:")
            logging.info(f"  数量变化: {before_count:,} → {after_count:,} ({count_change:+,})")
            logging.info(f"  占比变化: {before_pct:.2f}% → {after_pct:.2f}% ({pct_change:+.2f}%)")
        
        result['changes'] = changes
    
    # 分析产品名称分布的时间差异
    if 'Product Name' in df.columns:
        logging.info(f"\n--- 产品名称分布的时间差异 ---")
        
        if len(before_data) > 0 and len(after_data) > 0:
            # 按时间段和产品名称分组的OP-LASER分布
            before_product_laser = pd.crosstab(before_data['Product Name'], before_data['OP-LASER'])
            after_product_laser = pd.crosstab(after_data['Product Name'], after_data['OP-LASER'])
            
            logging.info(f"截止日期前产品配置分布:")
            logging.info(f"{'Product Name':<25} {'标准+Orin':>10} {'高阶+Thor':>10}")
            for product in before_product_laser.index:
                standard = before_product_laser.loc[product].get('标准+Orin', 0)
                thor = before_product_laser.loc[product].get('高阶+Thor', 0)
                logging.info(f"{product:<25} {standard:>10} {thor:>10}")
            
            logging.info(f"\n截止日期后产品配置分布:")
            logging.info(f"{'Product Name':<25} {'标准+Orin':>10} {'高阶+Thor':>10}")
            for product in after_product_laser.index:
                standard = after_product_laser.loc[product].get('标准+Orin', 0)
                thor = after_product_laser.loc[product].get('高阶+Thor', 0)
                logging.info(f"{product:<25} {standard:>10} {thor:>10}")
            
            result['product_time_analysis'] = {
                'before_product_distribution': before_product_laser.to_dict(),
                'after_product_distribution': after_product_laser.to_dict()
            }
    
    # 价格分析
    if '开票价格' in df.columns:
        logging.info(f"\n--- 价格趋势分析 ---")
        
        if len(before_data) > 0:
            before_price_stats = before_data.groupby('OP-LASER')['开票价格'].agg(['count', 'mean', 'median']).round(2)
            logging.info(f"截止日期前价格统计:\n{before_price_stats}")
            result['before_price_analysis'] = before_price_stats.to_dict()
        
        if len(after_data) > 0:
            after_price_stats = after_data.groupby('OP-LASER')['开票价格'].agg(['count', 'mean', 'median']).round(2)
            logging.info(f"\n截止日期后价格统计:\n{after_price_stats}")
            result['after_price_analysis'] = after_price_stats.to_dict()
    
    return result

def analyze_laser_configuration_with_xgboost(df, test_size=0.2, random_state=42):
    """
    使用XGBoost分析激光雷达配置对价格和销量的影响
    
    参数:
    - df: 数据框
    - test_size: 测试集比例
    - random_state: 随机种子
    
    返回:
    - 包含模型性能、特征重要性、SHAP值和情景预测的分析结果
    """
    if not ML_AVAILABLE:
        logging.error("机器学习库不可用，无法执行XGBoost分析")
        return None
    
    logging.info("=== XGBoost激光雷达配置影响分析 ===")
    
    # 数据预处理
    analysis_df = df.copy()
    
    # 移除缺失值过多的行
    analysis_df = analysis_df.dropna(subset=['OP-LASER', '开票价格'])
    
    if len(analysis_df) < 100:
        logging.warning("有效数据量不足，无法进行可靠的机器学习分析")
        return None
    
    logging.info(f"用于分析的数据量: {len(analysis_df)} 条")
    
    # 特征工程
    feature_columns = []
    
    # 1. 配置特征（二值化）
    config_features = ['OP-LASER', 'OP-FRIDGE', 'OP-LuxGift', 'OP-SW']
    for col in config_features:
        if col in analysis_df.columns:
            # 将配置转换为二值特征，特别处理OP-LASER
            if col == 'OP-LASER':
                # 对于OP-LASER，高阶+Thor为1，标准+Orin为0
                analysis_df[f'{col}_binary'] = analysis_df[col].apply(
                    lambda x: 1 if str(x) == '高阶+Thor' else 0
                )
            else:
                # 其他配置特征的二值化
                analysis_df[f'{col}_binary'] = analysis_df[col].apply(
                    lambda x: 1 if str(x) not in ['nan', 'None', '', 'NaN'] and pd.notna(x) else 0
                )
            feature_columns.append(f'{col}_binary')
    
    # 2. 分类特征（标签编码）
    categorical_features = ['EXCOLOR', 'INCOLOR', 'WHEEL', 'Product Name', 'Product_Types']
    label_encoders = {}
    
    for col in categorical_features:
        if col in analysis_df.columns:
            le = LabelEncoder()
            analysis_df[f'{col}_encoded'] = le.fit_transform(analysis_df[col].fillna('Unknown'))
            label_encoders[col] = le
            feature_columns.append(f'{col}_encoded')
    
    # 3. 时间特征
    if 'lock_time' in analysis_df.columns:
        analysis_df['lock_time'] = pd.to_datetime(analysis_df['lock_time'])
        analysis_df['lock_month'] = analysis_df['lock_time'].dt.month
        analysis_df['lock_day_of_week'] = analysis_df['lock_time'].dt.dayofweek
        feature_columns.extend(['lock_month', 'lock_day_of_week'])
    
    # 准备特征矩阵和目标变量
    X = analysis_df[feature_columns].fillna(0)
    y_price = analysis_df['开票价格']
    
    logging.info(f"特征数量: {len(feature_columns)}")
    logging.info(f"特征列表: {feature_columns}")
    
    # 分割数据
    X_train, X_test, y_train_price, y_test_price = train_test_split(
        X, y_price, test_size=test_size, random_state=random_state
    )
    
    results = {
        'analysis_time': datetime.now().isoformat(),
        'data_summary': {
            'total_samples': len(analysis_df),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(feature_columns)
        },
        'feature_columns': feature_columns
    }
    
    # 1. 价格预测模型
    logging.info("\n--- 训练价格预测模型 ---")
    
    price_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1
    )
    
    price_model.fit(X_train, y_train_price)
    
    # 模型评估
    y_pred_price = price_model.predict(X_test)
    price_mse = mean_squared_error(y_test_price, y_pred_price)
    price_r2 = r2_score(y_test_price, y_pred_price)
    
    logging.info(f"价格预测模型性能:")
    logging.info(f"  R² Score: {price_r2:.4f}")
    logging.info(f"  RMSE: {np.sqrt(price_mse):.2f}")
    
    results['price_model'] = {
        'r2_score': price_r2,
        'rmse': np.sqrt(price_mse),
        'feature_importance': dict(zip(feature_columns, price_model.feature_importances_))
    }
    
    # 2. SHAP分析
    logging.info("\n--- SHAP可解释性分析 ---")
    
    try:
        # 使用较小的样本进行SHAP分析以提高速度
        shap_sample_size = min(500, len(X_test))
        X_shap = X_test.iloc[:shap_sample_size]
        
        # 使用TreeExplainer专门处理XGBoost模型
        explainer = shap.TreeExplainer(price_model)
        shap_values = explainer.shap_values(X_shap)
        
        # 计算平均SHAP值
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        shap_importance = dict(zip(feature_columns, mean_shap_values))
        
        # 找到激光雷达相关特征的SHAP值
        laser_shap_value = shap_importance.get('OP-LASER_binary', 0)
        
        logging.info(f"激光雷达配置的平均SHAP影响值: {laser_shap_value:.2f}")
        
        results['shap_analysis'] = {
            'laser_shap_impact': laser_shap_value,
            'feature_shap_importance': shap_importance,
            'sample_size': shap_sample_size
        }
        
    except Exception as e:
        logging.warning(f"SHAP分析失败: {e}")
        results['shap_analysis'] = {'error': str(e)}
    
    # 3. 情景模拟：免费提供高阶+Thor的影响
    logging.info("\n--- 情景模拟：免费提供高阶+Thor ---")
    
    # 筛选标准配置的订单（OP-LASER = '标准+Orin'）
    standard_orders = analysis_df[analysis_df['OP-LASER'] == '标准+Orin'].copy()
    high_end_orders = analysis_df[analysis_df['OP-LASER'] == '高阶+Thor'].copy()
    
    if len(standard_orders) == 0:
        logging.info("没有找到标准配置的订单，无法进行情景模拟")
        results['scenario_simulation'] = {'error': '没有标准配置订单'}
    else:
        # 分析当前配置价格差异
        avg_standard_price = standard_orders['开票价格'].mean()
        avg_high_end_price = high_end_orders['开票价格'].mean() if len(high_end_orders) > 0 else avg_standard_price
        
        # 计算配置价值差异（用于成本分析）
        config_value_difference = avg_high_end_price - avg_standard_price
        
        # 免费提供情景：价格保持不变，但产品价值提升
        # 计算价值提升对销量的正面影响
        
        # 方法1：基于产品价值提升的需求弹性
        # 假设免费获得高端配置会提升产品吸引力，增加销量
        value_elasticity = 0.8  # 价值弹性系数：价值提升1%，销量增长0.8%
        value_increase_ratio = config_value_difference / avg_standard_price
        volume_increase_from_value = value_elasticity * value_increase_ratio
        
        # 方法2：基于竞争力提升的市场份额增长
        # 假设免费高端配置提升竞争力，从竞争对手获得市场份额
        competitive_advantage_boost = 0.15  # 假设竞争力提升带来15%的销量增长
        
        # 综合销量影响（取较保守的估计）
        total_volume_change = min(volume_increase_from_value, competitive_advantage_boost)
        
        # 成本影响分析
        total_cost_increase = len(standard_orders) * config_value_difference
        
        logging.info(f"=== 免费提供高阶+Thor配置分析 ===")
        logging.info(f"标准配置订单数量: {len(standard_orders):,}")
        logging.info(f"高阶配置订单数量: {len(high_end_orders):,}")
        logging.info(f"标准配置平均价格: {avg_standard_price:,.2f} 元")
        logging.info(f"高阶配置平均价格: {avg_high_end_price:,.2f} 元")
        logging.info(f"配置价值差异: {config_value_difference:,.2f} 元")
        logging.info(f"")
        logging.info(f"=== 情景模拟结果 ===")
        logging.info(f"客户支付价格: 保持不变 (免费提供)")
        logging.info(f"产品价值提升: {value_increase_ratio:.2%}")
        logging.info(f"预估销量增长: +{total_volume_change:.2%}")
        logging.info(f"单车成本增加: {config_value_difference:,.2f} 元")
        logging.info(f"总成本增加: {total_cost_increase/1e8:.2f} 亿元")
        
        results['scenario_simulation'] = {
            'scenario_type': '免费提供高阶配置',
            'standard_orders_count': len(standard_orders),
            'high_end_orders_count': len(high_end_orders),
            'standard_avg_price': float(avg_standard_price),
            'high_end_avg_price': float(avg_high_end_price),
            'config_value_difference': float(config_value_difference),
            'customer_price_change': 0.0,  # 免费提供，客户价格不变
            'product_value_increase_ratio': float(value_increase_ratio),
            'estimated_volume_change': float(total_volume_change),
            'unit_cost_increase': float(config_value_difference),
            'total_cost_increase': float(total_cost_increase),
            'assumptions': {
                'value_elasticity': value_elasticity,
                'competitive_advantage_boost': competitive_advantage_boost,
                'description': '基于产品价值提升和竞争力增强的销量影响估算'
            }
        }
    
    # 4. 特征重要性排序
    feature_importance_sorted = sorted(
        results['price_model']['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    logging.info(f"\n--- 特征重要性排序（Top 10）---")
    for i, (feature, importance) in enumerate(feature_importance_sorted[:10]):
        logging.info(f"  {i+1}. {feature}: {importance:.4f}")
    
    results['top_features'] = feature_importance_sorted[:10]
    
    return results

def analyze_all_configuration_fields(df):
    """分析所有配置字段"""
    logging.info("=== 分析所有配置字段 ===")
    
    # 识别配置字段（排除基础信息字段）
    base_fields = ['lock_time', 'invoice_time', 'order_number', 'Product Name', 'Product_Types', '开票价格']
    config_fields = [col for col in df.columns if col not in base_fields]
    
    logging.info(f"识别到的配置字段: {config_fields}")
    
    results = {}
    for field in config_fields:
        logging.info(f"\n--- 分析字段: {field} ---")
        field_result = analyze_field_values(df, field)
        if field_result:
            results[field] = field_result
    
    return results

def generate_summary_report(df, analysis_results):
    """生成分析摘要报告"""
    logging.info("=== 生成分析摘要报告 ===")
    
    report = {
        'analysis_time': datetime.now().isoformat(),
        'data_summary': {
            'total_orders': len(df),
            'date_range': {
                'earliest_lock': df['lock_time'].min().isoformat() if 'lock_time' in df.columns else None,
                'latest_lock': df['lock_time'].max().isoformat() if 'lock_time' in df.columns else None
            },
            'price_range': {
                'min_price': float(df['开票价格'].min()) if '开票价格' in df.columns else None,
                'max_price': float(df['开票价格'].max()) if '开票价格' in df.columns else None,
                'avg_price': float(df['开票价格'].mean()) if '开票价格' in df.columns else None
            }
        },
        'configuration_analysis': analysis_results
    }
    
    return report

def save_analysis_results(results, output_path):
    """保存分析结果"""
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logging.info(f"分析结果已保存到: {output_path}")
        
    except Exception as e:
        logging.error(f"保存分析结果失败: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CM2激光雷达配置(OP-LASER)专项分析脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 默认激光雷达配置分析（推荐）
  python cm2_configuration_OP-LASER.py -i data.csv
  
  # 专门分析激光雷达配置（详细模式）
  python cm2_configuration_OP-LASER.py -i data.csv --laser-analysis
  
  # 分析OP-LASER字段的基础统计
  python cm2_configuration_OP-LASER.py -i data.csv --field OP-LASER
  
  # 生成产品名称与激光雷达配置的交叉统计表
  python cm2_configuration_OP-LASER.py -i data.csv --cross-analysis "Product Name" "OP-LASER"
  
  # 分析指定日期前后的激光雷达配置变化趋势
  python cm2_configuration_OP-LASER.py -i data.csv --time-analysis "2025-10-15"
  
  # 使用XGBoost机器学习分析激光雷达配置影响（推荐）
  python cm2_configuration_OP-LASER.py -i data.csv --xgboost-analysis
  
  # 生成产品类型与激光雷达配置的交叉分析
  python cm2_configuration_OP-LASER.py -i data.csv --cross-analysis "Product_Types" "OP-LASER"
  
  # 保存XGBoost分析结果到JSON文件
  python cm2_configuration_OP-LASER.py -i data.csv --xgboost-analysis -o xgboost_analysis_report.json
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入的转置后CM2配置数据CSV文件路径'
    )
    
    parser.add_argument(
        '--field',
        help='指定要分析的字段名'
    )
    
    parser.add_argument(
        '--all-fields',
        action='store_true',
        help='分析所有配置字段'
    )
    
    parser.add_argument(
        '--laser-analysis',
        action='store_true',
        help='专门分析激光雷达配置'
    )
    
    parser.add_argument(
        '--cross-analysis',
        nargs=2,
        metavar=('ROW_FIELD', 'COL_FIELD'),
        help='生成两个字段的交叉统计表，格式: --cross-analysis "Product Name" "OP-LASER"'
    )
    
    parser.add_argument(
        '--time-analysis',
        metavar='CUTOFF_DATE',
        help='分析指定日期前后的OP-LASER配置差异，格式: --time-analysis "2025-10-15"'
    )
    
    parser.add_argument(
        '--xgboost-analysis',
        action='store_true',
        help='使用XGBoost机器学习分析激光雷达配置对价格和销量的影响，包括SHAP解释和情景模拟'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出分析结果的JSON文件路径（可选）'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别（默认: INFO）'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        # 加载数据
        df = load_transposed_data(args.input)
        
        analysis_results = {}
        
        # 根据参数执行相应的分析
        if args.field:
            # 分析指定字段
            result = analyze_field_values(df, args.field)
            if result:
                analysis_results[args.field] = result
        
        elif args.cross_analysis:
            # 生成交叉统计表
            row_field, col_field = args.cross_analysis
            cross_tab = generate_cross_tabulation_report(
                df, row_field, col_field, 
                f"{row_field} 与 {col_field} 交叉统计分析"
            )
            if cross_tab is not None:
                analysis_results[f'{row_field}_vs_{col_field}'] = {
                    'cross_tabulation': cross_tab.to_dict(),
                    'row_field': row_field,
                    'col_field': col_field
                }
        
        elif args.time_analysis:
            # 分析指定日期前后的OP-LASER配置差异
            result = analyze_time_based_laser_configuration(df, args.time_analysis)
            if result:
                analysis_results[f'time_analysis_{args.time_analysis}'] = result
        
        elif args.xgboost_analysis:
            # 使用XGBoost机器学习分析激光雷达配置影响
            result = analyze_laser_configuration_with_xgboost(df)
            if result:
                analysis_results['xgboost_laser_analysis'] = result
        
        elif args.laser_analysis:
            # 专门分析激光雷达配置
            result = analyze_laser_configuration(df)
            if result:
                analysis_results['OP-LASER'] = result
        
        elif args.all_fields:
            # 分析所有配置字段
            analysis_results = analyze_all_configuration_fields(df)
        
        else:
            # 默认分析激光雷达配置
            logging.info("未指定具体分析内容，默认分析激光雷达配置")
            result = analyze_laser_configuration(df)
            if result:
                analysis_results['OP-LASER'] = result
        
        # 生成摘要报告
        if analysis_results:
            summary_report = generate_summary_report(df, analysis_results)
            
            # 保存结果（如果指定了输出路径）
            if args.output:
                save_analysis_results(summary_report, args.output)
            
            logging.info("=== 分析完成 ===")
        else:
            logging.warning("没有生成任何分析结果")
    
    except Exception as e:
        logging.error(f"分析过程中发生错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()