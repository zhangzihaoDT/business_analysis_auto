#!/usr/bin/env python3
"""
CM2配置项影响分析建模脚本

功能：
1. 数据筛选：lock_time >= 2025-09-10 且 invoice_time 不为空
2. 线性回归和XGBoost回归分析配置项对开票价格的影响
3. 分析配置项对销量的影响
4. 生成详细的分析报告和可视化结果

作者：AI Assistant
创建时间：2025-10-23
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings
from datetime import datetime
import argparse
import os
import sys
from pathlib import Path

# 机器学习相关库
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# 设置Plotly默认主题和中文支持
pio.templates.default = "plotly_white"
warnings.filterwarnings('ignore')

class CM2ConfigurationAnalyzer:
    """CM2配置项影响分析器"""
    
    def __init__(self, data_path, output_dir=None):
        """
        初始化分析器
        
        Args:
            data_path (str): 数据文件路径
            output_dir (str): 输出目录路径
        """
        self.data_path = data_path
        self.output_dir = output_dir or os.path.join(os.path.dirname(data_path), 'analysis_results')
        self.raw_data = None
        self.filtered_data = None
        self.feature_columns = ['Product Name', 'Product_Types', 'EXCOLOR', 'INCOLOR', 'OP-FRIDGE', 'OP-LASER', 'OP-LuxGift', 'OP-SW', 'WHEEL']
        self.target_column = '开票价格'
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"配置项影响分析器初始化完成")
        print(f"数据文件: {self.data_path}")
        print(f"输出目录: {self.output_dir}")
    
    def load_and_filter_data(self):
        """加载并筛选数据"""
        print("\n=== 数据加载和筛选 ===")
        
        # 加载数据
        print(f"正在加载数据: {self.data_path}")
        self.raw_data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {self.raw_data.shape}")
        
        # 转换日期字段
        self.raw_data['lock_time'] = pd.to_datetime(self.raw_data['lock_time'])
        
        # 数据筛选
        print("应用筛选条件:")
        print("1. lock_time >= 2025-09-10")
        print("2. invoice_time 不为空")
        
        # 筛选条件
        condition = (
            (self.raw_data['lock_time'] >= '2025-09-10') &
            (self.raw_data['invoice_time'].notna())
        )
        
        self.filtered_data = self.raw_data[condition].copy()
        print(f"筛选后数据形状: {self.filtered_data.shape}")
        print(f"筛选保留比例: {len(self.filtered_data)/len(self.raw_data)*100:.2f}%")
        
        # 数据基本信息
        print(f"\n筛选后数据概览:")
        print(f"- 时间范围: {self.filtered_data['lock_time'].min()} 至 {self.filtered_data['lock_time'].max()}")
        print(f"- 开票价格范围: {self.filtered_data[self.target_column].min():,.0f} - {self.filtered_data[self.target_column].max():,.0f}")
        print(f"- 不同订单数量: {self.filtered_data['order_number'].nunique():,}")
        
        return self.filtered_data
    
    def prepare_features(self):
        """准备特征数据"""
        print("\n=== 特征数据准备 ===")
        
        # 检查特征列
        missing_features = [col for col in self.feature_columns if col not in self.filtered_data.columns]
        if missing_features:
            raise ValueError(f"缺少特征列: {missing_features}")
        
        # 创建特征数据副本
        feature_data = self.filtered_data[self.feature_columns + [self.target_column]].copy()
        
        # 处理分类特征
        label_encoders = {}
        for col in self.feature_columns:
            if feature_data[col].dtype == 'object':
                le = LabelEncoder()
                feature_data[f'{col}_encoded'] = le.fit_transform(feature_data[col].astype(str))
                label_encoders[col] = le
                print(f"对 {col} 进行标签编码，类别数: {len(le.classes_)}")
            else:
                feature_data[f'{col}_encoded'] = feature_data[col]
                print(f"{col} 已为数值类型，无需编码")
        
        # 获取编码后的特征列
        encoded_features = [f'{col}_encoded' for col in self.feature_columns]
        
        # 特征统计
        print(f"\n特征统计:")
        for col in self.feature_columns:
            print(f"- {col}: {feature_data[col].value_counts().to_dict()}")
        
        self.feature_data = feature_data
        self.encoded_features = encoded_features
        self.label_encoders = label_encoders
        
        return feature_data, encoded_features
    
    def linear_regression_analysis(self):
        """线性回归分析"""
        print("\n=== 线性回归分析 ===")
        
        # 准备数据
        X = self.feature_data[self.encoded_features]
        y = self.feature_data[self.target_column]
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = lr_model.predict(X_train)
        y_pred_test = lr_model.predict(X_test)
        
        # 评估指标
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"线性回归模型性能:")
        print(f"训练集 R²: {train_r2:.4f}")
        print(f"测试集 R²: {test_r2:.4f}")
        print(f"训练集 RMSE: {train_rmse:,.0f}")
        print(f"测试集 RMSE: {test_rmse:,.0f}")
        print(f"训练集 MAE: {train_mae:,.0f}")
        print(f"测试集 MAE: {test_mae:,.0f}")
        
        # 特征重要性（系数）
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': lr_model.coef_,
            'abs_coefficient': np.abs(lr_model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        print(f"\n线性回归特征重要性（系数）:")
        for _, row in feature_importance.iterrows():
            print(f"- {row['feature']}: {row['coefficient']:,.2f}")
        
        # 交叉验证
        cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='r2')
        print(f"\n5折交叉验证 R² 分数: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # 保存结果
        self.lr_results = {
            'model': lr_model,
            'scaler': scaler,
            'feature_importance': feature_importance,
            'metrics': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            },
            'predictions': {
                'y_train': y_train,
                'y_test': y_test,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }
        }
        
        return self.lr_results
    
    def xgboost_regression_analysis(self):
        """XGBoost回归分析"""
        print("\n=== XGBoost回归分析 ===")
        
        # 准备数据
        X = self.feature_data[self.encoded_features]
        y = self.feature_data[self.target_column]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练XGBoost模型
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = xgb_model.predict(X_train)
        y_pred_test = xgb_model.predict(X_test)
        
        # 评估指标
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"XGBoost模型性能:")
        print(f"训练集 R²: {train_r2:.4f}")
        print(f"测试集 R²: {test_r2:.4f}")
        print(f"训练集 RMSE: {train_rmse:,.0f}")
        print(f"测试集 RMSE: {test_rmse:,.0f}")
        print(f"训练集 MAE: {train_mae:,.0f}")
        print(f"测试集 MAE: {test_mae:,.0f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nXGBoost特征重要性:")
        for _, row in feature_importance.iterrows():
            print(f"- {row['feature']}: {row['importance']:.4f}")
        
        # 交叉验证
        cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
        print(f"\n5折交叉验证 R² 分数: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # 保存结果
        self.xgb_results = {
            'model': xgb_model,
            'feature_importance': feature_importance,
            'metrics': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            },
            'predictions': {
                'y_train': y_train,
                'y_test': y_test,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }
        }
        
        return self.xgb_results
    
    def sales_volume_analysis(self):
        """销量分析（按配置项统计订单数量）"""
        print("\n=== 销量分析 ===")
        
        # 总体销量
        total_orders = len(self.filtered_data)
        unique_orders = self.filtered_data['order_number'].nunique()
        print(f"总订单记录数: {total_orders:,}")
        print(f"唯一订单数: {unique_orders:,}")
        
        # 按配置项分析销量
        sales_analysis = {}
        
        for feature in self.feature_columns:
            print(f"\n--- {feature} 销量分析 ---")
            
            # 按配置项值统计订单数量
            feature_sales = self.filtered_data.groupby(feature).agg({
                'order_number': 'count',  # 订单记录数
                'order_number': 'nunique'  # 唯一订单数
            }).rename(columns={'order_number': 'unique_orders'})
            
            # 重新计算（因为上面的聚合有问题）
            feature_sales = self.filtered_data.groupby(feature).agg({
                'order_number': ['count', 'nunique'],
                self.target_column: ['mean', 'median', 'std']
            })
            
            # 扁平化列名
            feature_sales.columns = ['_'.join(col).strip() for col in feature_sales.columns]
            feature_sales = feature_sales.rename(columns={
                'order_number_count': 'total_records',
                'order_number_nunique': 'unique_orders',
                f'{self.target_column}_mean': 'avg_price',
                f'{self.target_column}_median': 'median_price',
                f'{self.target_column}_std': 'price_std'
            })
            
            # 计算市场份额
            feature_sales['market_share'] = feature_sales['unique_orders'] / unique_orders * 100
            
            # 排序
            feature_sales = feature_sales.sort_values('unique_orders', ascending=False)
            
            print(feature_sales.round(2))
            
            sales_analysis[feature] = feature_sales
        
        self.sales_analysis = sales_analysis
        return sales_analysis
    
    def calculate_configuration_contribution(self):
        """计算配置项对成交价和销量的贡献"""
        print("\n=== 计算配置收益贡献 ===")
        
        contribution_data = []
        
        for feature in self.feature_columns:
            # 获取线性回归系数（价格贡献）
            lr_coef = self.lr_results['feature_importance'][
                self.lr_results['feature_importance']['feature'] == feature
            ]['coefficient'].iloc[0]
            
            # 获取XGBoost重要性
            xgb_importance = self.xgb_results['feature_importance'][
                self.xgb_results['feature_importance']['feature'] == feature
            ]['importance'].iloc[0]
            
            # 计算销量影响（基于不同配置选项的市场份额差异）
            sales_data = self.sales_analysis[feature]
            if len(sales_data) > 1:
                # 计算最高份额与最低份额的差异
                max_share = sales_data['market_share'].max()
                min_share = sales_data['market_share'].min()
                sales_impact = max_share - min_share
                
                # 计算平均价格差异
                max_price_option = sales_data.loc[sales_data['avg_price'].idxmax()]
                min_price_option = sales_data.loc[sales_data['avg_price'].idxmin()]
                price_diff = max_price_option['avg_price'] - min_price_option['avg_price']
            else:
                sales_impact = 0
                price_diff = 0
            
            # 生成结论
            if lr_coef > 5000 and sales_impact > 10:
                conclusion = "明显受欢迎，建议标配"
            elif lr_coef > 3000 and sales_impact > 5:
                conclusion = "稳定增益项"
            elif lr_coef > 1000 and sales_impact < 5:
                conclusion = "提价多但不带动销量"
            elif lr_coef < 1000 and sales_impact > 10:
                conclusion = "强拉动，建议优化续航" if "电池" in feature else "强拉动，建议推广"
            elif abs(lr_coef) < 500 and abs(sales_impact) < 2:
                conclusion = "中性配置，可视成本优化"
            elif lr_coef > 0 and sales_impact > 0:
                conclusion = "弱正向，可留在高配"
            else:
                conclusion = "影响较小，可考虑简化"
            
            contribution_data.append({
                'feature': feature,
                'price_contribution': lr_coef / 10000,  # 转换为万元
                'sales_impact': sales_impact,
                'xgb_importance': xgb_importance,
                'conclusion': conclusion,
                'price_diff': price_diff
            })
        
        self.contribution_df = pd.DataFrame(contribution_data).sort_values(
            'xgb_importance', ascending=False
        )
        
        return self.contribution_df
    
    def generate_price_elasticity_analysis(self):
        """生成价格-销量响应曲线分析"""
        print("\n=== 价格弹性分析 ===")
        
        # 基于现有数据生成不同配置组合的价格-销量预测
        base_price = self.filtered_data[self.target_column].median()
        
        # 定义几种典型配置组合
        config_scenarios = {
            '基础版': {'OP-FRIDGE': 0, 'WHEEL': 0, 'OP-LuxGift': 0},
            '舒适版': {'OP-FRIDGE': 0, 'WHEEL': 1, 'OP-LuxGift': 1},
            '豪华版': {'OP-FRIDGE': 1, 'WHEEL': 0, 'OP-LuxGift': 1},
            '旗舰版': {'OP-FRIDGE': 1, 'WHEEL': 1, 'OP-LuxGift': 1}
        }
        
        elasticity_data = []
        
        for scenario_name, config in config_scenarios.items():
            # 计算该配置的预期价格
            price_adjustment = 0
            for feature, value in config.items():
                if feature in self.feature_columns:
                    feature_coef = self.lr_results['feature_importance'][
                        self.lr_results['feature_importance']['feature'] == feature
                    ]['coefficient'].iloc[0]
                    price_adjustment += feature_coef * value
            
            predicted_price = base_price + price_adjustment
            
            # 基于价格预测销量（简化的弹性模型）
            # 假设价格弹性系数为-1.5（价格每增加1%，销量减少1.5%）
            price_change_pct = (predicted_price - base_price) / base_price
            volume_change_pct = -1.5 * price_change_pct
            base_volume = len(self.filtered_data)
            predicted_volume = base_volume * (1 + volume_change_pct)
            
            elasticity_data.append({
                'scenario': scenario_name,
                'price': predicted_price / 10000,  # 转换为万元
                'volume': max(0, predicted_volume),
                'config': config
            })
        
        self.elasticity_df = pd.DataFrame(elasticity_data)
        return self.elasticity_df
    
    def generate_visualizations(self):
        """生成Plotly可视化图表"""
        print("\n=== 生成可视化图表 ===")
        
        # 1. 特征重要性对比图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('线性回归特征重要性（绝对系数值）', 'XGBoost特征重要性'),
            horizontal_spacing=0.1
        )
        
        # 线性回归特征重要性
        lr_importance = self.lr_results['feature_importance'].copy()
        fig.add_trace(
            go.Bar(
                y=lr_importance['feature'],
                x=lr_importance['abs_coefficient'],
                orientation='h',
                name='线性回归',
                marker_color='lightblue',
                text=[f'{x:.0f}' for x in lr_importance['abs_coefficient']],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # XGBoost特征重要性
        xgb_importance = self.xgb_results['feature_importance'].copy()
        fig.add_trace(
            go.Bar(
                y=xgb_importance['feature'],
                x=xgb_importance['importance'],
                orientation='h',
                name='XGBoost',
                marker_color='lightcoral',
                text=[f'{x:.3f}' for x in xgb_importance['importance']],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="特征重要性对比分析",
            height=600,
            showlegend=False,
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        fig.update_xaxes(title_text="绝对系数值", row=1, col=1)
        fig.update_xaxes(title_text="重要性分数", row=1, col=2)
        
        fig.write_html(os.path.join(self.output_dir, 'feature_importance_comparison.html'))
        fig.write_image(os.path.join(self.output_dir, 'feature_importance_comparison.png'))
        
        # 2. 配置收益贡献可视化
        contrib_df = self.contribution_df
        
        fig = go.Figure()
        
        # 创建气泡图：x轴为价格贡献，y轴为销量影响，气泡大小为XGBoost重要性
        fig.add_trace(go.Scatter(
            x=contrib_df['price_contribution'],
            y=contrib_df['sales_impact'],
            mode='markers+text',
            marker=dict(
                size=contrib_df['xgb_importance'] * 1000,  # 放大气泡
                color=contrib_df['price_contribution'],
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title="价格贡献(万元)")
            ),
            text=contrib_df['feature'],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hovertemplate='<b>%{text}</b><br>' +
                         '价格贡献: %{x:.2f}万元<br>' +
                         '销量影响: %{y:.1f}%<br>' +
                         '重要性: %{marker.size:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="配置项收益贡献分析",
            xaxis_title="对成交价贡献（万元）",
            yaxis_title="对销量影响（%）",
            height=600,
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        # 添加象限分割线
        fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0.3, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.write_html(os.path.join(self.output_dir, 'configuration_contribution_analysis.html'))
        fig.write_image(os.path.join(self.output_dir, 'configuration_contribution_analysis.png'))
        
        # 3. 价格-销量响应曲线
        elasticity_df = self.elasticity_df
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (scenario, data) in enumerate(elasticity_df.groupby('scenario')):
            fig.add_trace(go.Scatter(
                x=data['price'],
                y=data['volume'],
                mode='markers+lines',
                name=scenario,
                marker=dict(size=12, color=colors[i % len(colors)]),
                line=dict(width=3, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title="价格-销量响应曲线（Price Elasticity Curve）",
            xaxis_title="售价（万元）",
            yaxis_title="预测销量",
            height=600,
            font=dict(family="Arial, sans-serif", size=12),
            hovermode='x unified'
        )
        
        fig.write_html(os.path.join(self.output_dir, 'price_elasticity_curve.html'))
        fig.write_image(os.path.join(self.output_dir, 'price_elasticity_curve.png'))
        
        # 4. 销量分析热力图
        sales_summary = []
        for feature in self.feature_columns:
            sales_data = self.sales_analysis[feature]
            for option, data in sales_data.iterrows():
                sales_summary.append({
                    'feature': feature,
                    'option': str(option),
                    'orders': data['unique_orders'],
                    'market_share': data['market_share'],
                    'avg_price': data['avg_price']
                })
        
        sales_df = pd.DataFrame(sales_summary)
        
        # 创建热力图数据
        pivot_data = sales_df.pivot_table(
            index='feature', 
            columns='option', 
            values='market_share', 
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Blues',
            text=[[f'{val:.1f}%' for val in row] for row in pivot_data.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="市场份额(%)")
        ))
        
        fig.update_layout(
            title="配置选项市场份额热力图",
            xaxis_title="配置选项",
            yaxis_title="配置项",
            height=600,
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        fig.write_html(os.path.join(self.output_dir, 'sales_heatmap.html'))
        fig.write_image(os.path.join(self.output_dir, 'sales_heatmap.png'))
        
        print(f"可视化图表已保存到: {self.output_dir}")
        print("生成的图表文件:")
        print("- feature_importance_comparison.html/png")
        print("- configuration_contribution_analysis.html/png") 
        print("- price_elasticity_curve.html/png")
        print("- sales_heatmap.html/png")
    
    def generate_report(self):
        """生成专业分析报告"""
        print("\n=== 生成分析报告 ===")
        
        report_path = os.path.join(self.output_dir, 'cm2_configuration_impact_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🚗 CM2配置项收益贡献分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # 数据概览
            f.write("## 📊 一、数据概览\n\n")
            f.write(f"- **原始数据量**: {len(self.raw_data):,} 条记录\n")
            f.write(f"- **过滤后数据量**: {len(self.filtered_data):,} 条记录\n")
            f.write(f"- **数据保留率**: {len(self.filtered_data)/len(self.raw_data)*100:.2f}%\n")
            f.write(f"- **唯一订单数**: {self.filtered_data['order_number'].nunique():,} 个\n")
            f.write(f"- **价格范围**: {self.filtered_data[self.target_column].min()/10000:.1f} - {self.filtered_data[self.target_column].max()/10000:.1f} 万元\n")
            f.write(f"- **平均价格**: {self.filtered_data[self.target_column].mean()/10000:.1f} 万元\n\n")
            
            # 模型性能对比
            f.write("## 🎯 二、模型性能对比\n\n")
            f.write("| 模型 | 训练集R² | 测试集R² | RMSE | MAE | 交叉验证R² |\n")
            f.write("|------|----------|----------|------|-----|------------|\n")
            
            lr_metrics = self.lr_results['metrics']
            f.write(f"| 线性回归 | {lr_metrics['train_r2']:.4f} | {lr_metrics['test_r2']:.4f} | "
                   f"{lr_metrics['test_rmse']/10000:.2f}万 | {lr_metrics['test_mae']/10000:.2f}万 | {lr_metrics['cv_r2_mean']:.4f} |\n")
            
            xgb_metrics = self.xgb_results['metrics']
            f.write(f"| XGBoost | {xgb_metrics['train_r2']:.4f} | {xgb_metrics['test_r2']:.4f} | "
                   f"{xgb_metrics['test_rmse']/10000:.2f}万 | {xgb_metrics['test_mae']/10000:.2f}万 | {xgb_metrics['cv_r2_mean']:.4f} |\n\n")
            
            f.write("> 💡 **模型选择建议**: ")
            if xgb_metrics['test_r2'] > lr_metrics['test_r2']:
                f.write(f"XGBoost模型表现更优（R² {xgb_metrics['test_r2']:.4f}），建议用于价格预测\n\n")
            else:
                f.write(f"线性回归模型表现更优（R² {lr_metrics['test_r2']:.4f}），建议用于价格预测\n\n")
            
            f.write("---\n\n")
            
            # 核心分析结果
            f.write("## 🧭 三、核心分析结果\n\n")
            
            # 配置收益贡献表
            f.write("### 1️⃣ 配置收益贡献表\n\n")
            f.write("| 配置项 | 对成交价贡献（万元） | 对销量影响（%） | 结论 |\n")
            f.write("|--------|---------------------|----------------|------|\n")
            
            for _, row in self.contribution_df.iterrows():
                f.write(f"| {row['feature']} | {row['price_contribution']:+.1f} | {row['sales_impact']:+.1f}% | {row['conclusion']} |\n")
            
            f.write("\n> 💡 **这张表最能打动决策者**：哪些配置\"值得加钱\"，哪些\"消费者不在意\"。\n\n")
            
            f.write("---\n\n")
            
            # 价格-销量响应曲线
            f.write("### 2️⃣ 价格-销量响应曲线（Price Elasticity Curve）\n\n")
            f.write("通过模拟不同配置组合的价格 → 销量曲线，\n")
            f.write("可以得到每个配置包的\"性价比最优点\"。\n\n")
            
            f.write("| 配置方案 | 预测售价（万元） | 预测销量 | 性价比评分 |\n")
            f.write("|----------|------------------|----------|------------|\n")
            
            for _, row in self.elasticity_df.iterrows():
                # 计算性价比评分（销量/价格的归一化值）
                performance_score = (row['volume'] / row['price']) / 1000
                f.write(f"| {row['scenario']} | {row['price']:.1f} | {row['volume']:.0f} | {performance_score:.2f} |\n")
            
            f.write("\n```\nX轴：售价（万元）\nY轴：销量预测\n不同颜色：配置组合方案\n```\n\n")
            f.write("👉 **帮助回答**：\"旗舰版再加智能驾驶包，销量会不会掉太多？\"\n\n")
            
            f.write("---\n\n")
            
            # 详细销量分析
            f.write("## 📈 四、详细销量分析\n\n")
            
            for feature in self.feature_columns:
                f.write(f"### {feature}\n\n")
                sales_data = self.sales_analysis[feature]
                
                f.write("| 配置选项 | 订单数量 | 市场份额 | 平均价格 | 中位价格 |\n")
                f.write("|----------|----------|----------|----------|----------|\n")
                
                for option, data in sales_data.iterrows():
                    f.write(f"| {option} | {data['unique_orders']:,} | {data['market_share']:.1f}% | "
                           f"{data['avg_price']/10000:.1f}万 | {data['median_price']/10000:.1f}万 |\n")
                f.write("\n")
            
            f.write("---\n\n")
            
            # 商业洞察和建议
            f.write("## 💡 五、商业洞察和建议\n\n")
            
            f.write("### 🎯 关键发现\n\n")
            
            # 找出最重要的配置项
            top_feature = self.contribution_df.iloc[0]
            f.write(f"1. **最重要配置项**: {top_feature['feature']} \n")
            f.write(f"   - 价格贡献: {top_feature['price_contribution']:+.1f}万元\n")
            f.write(f"   - 销量影响: {top_feature['sales_impact']:+.1f}%\n")
            f.write(f"   - 建议: {top_feature['conclusion']}\n\n")
            
            # 找出最受欢迎的配置
            f.write("2. **热门配置组合**:\n")
            for feature in self.feature_columns[:3]:  # 只显示前3个
                sales_data = self.sales_analysis[feature]
                top_option = sales_data.loc[sales_data['market_share'].idxmax()]
                f.write(f"   - {feature}: {top_option.name} (市场份额 {top_option['market_share']:.1f}%)\n")
            f.write("\n")
            
            # 价格弹性洞察
            best_scenario = self.elasticity_df.loc[self.elasticity_df['volume'].idxmax()]
            f.write(f"3. **最优性价比方案**: {best_scenario['scenario']}\n")
            f.write(f"   - 预测价格: {best_scenario['price']:.1f}万元\n")
            f.write(f"   - 预测销量: {best_scenario['volume']:.0f}台\n\n")
            
            f.write("### 🚀 行动建议\n\n")
            f.write("#### 短期策略（1-3个月）\n")
            f.write("1. **定价优化**: 重点调整高贡献配置项的价格梯度\n")
            f.write("2. **库存调整**: 根据市场份额数据优化各配置选项库存比例\n")
            f.write("3. **营销重点**: 突出推广高价值、高接受度的配置组合\n\n")
            
            f.write("#### 中期策略（3-6个月）\n")
            f.write("1. **产品组合优化**: 考虑将热门配置标准化到中高配车型\n")
            f.write("2. **价格策略调整**: 基于弹性分析结果调整不同配置方案定价\n")
            f.write("3. **客户细分**: 针对不同价格敏感度客户推荐合适配置\n\n")
            
            f.write("#### 长期策略（6个月以上）\n")
            f.write("1. **产品规划**: 基于配置贡献分析指导下一代产品开发\n")
            f.write("2. **供应链优化**: 调整高价值配置的供应商合作策略\n")
            f.write("3. **品牌定位**: 强化高贡献配置项的品牌价值传播\n\n")
            
            f.write("---\n\n")
            f.write("**报告说明**: 本分析基于历史销售数据，建议结合市场调研和竞品分析进行决策。\n")
        
        print(f"专业分析报告已保存到: {report_path}")
        
        # 保存数据结果
        results_data = {
            'linear_regression_importance': self.lr_results['feature_importance'],
            'xgboost_importance': self.xgb_results['feature_importance'],
            'sales_analysis': self.sales_analysis
        }
        
        # 保存为Excel文件
        excel_path = os.path.join(self.output_dir, 'cm2_configuration_analysis_results.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            results_data['linear_regression_importance'].to_excel(writer, sheet_name='线性回归重要性', index=False)
            results_data['xgboost_importance'].to_excel(writer, sheet_name='XGBoost重要性', index=False)
            
            for feature, data in results_data['sales_analysis'].items():
                data.to_excel(writer, sheet_name=f'{feature}_销量分析')
        
        print(f"数据结果已保存到: {excel_path}")
        
        return report_path, excel_path
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("开始CM2配置项影响分析...")
        
        try:
            # 1. 数据加载和筛选
            self.load_and_filter_data()
            
            # 2. 特征准备
            self.prepare_features()
            
            # 3. 线性回归分析
            self.linear_regression_analysis()
            
            # 4. XGBoost回归分析
            self.xgboost_regression_analysis()
            
            # 5. 销量分析
            self.sales_volume_analysis()
            
            # 6. 计算配置收益贡献
            self.calculate_configuration_contribution()
            
            # 7. 生成价格弹性分析
            self.generate_price_elasticity_analysis()
            
            # 8. 生成可视化
            self.generate_visualizations()
            
            # 9. 生成报告
            report_path, excel_path = self.generate_report()
            
            print(f"\n=== 分析完成 ===")
            print(f"分析报告: {report_path}")
            print(f"数据结果: {excel_path}")
            print(f"图表文件: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CM2配置项影响分析')
    parser.add_argument('-i', '--input', required=True, help='输入CSV文件路径')
    parser.add_argument('-o', '--output', help='输出目录路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 - {args.input}")
        sys.exit(1)
    
    # 创建分析器
    analyzer = CM2ConfigurationAnalyzer(args.input, args.output)
    
    # 运行分析
    success = analyzer.run_full_analysis()
    
    if success:
        print("分析成功完成！")
        sys.exit(0)
    else:
        print("分析失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()