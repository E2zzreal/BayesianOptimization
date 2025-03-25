import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel

class BayesianOptimizer:
    """
    贝叶斯优化器类，用于特征空间搜索和实验推荐
    """
    
    def __init__(self, model=None, feature_ranges=None, acquisition_function='ei', maximize=True, random_state=42):
        """
        初始化贝叶斯优化器
        
        参数:
            model: 训练好的模型，用于预测
            feature_ranges: 特征范围字典，格式为 {feature_name: {'min': min_val, 'max': max_val, 'step': step_val}}
            acquisition_function: 采样函数，可选 'ei'(期望改进), 'ucb'(置信上界), 'pi'(改进概率)
            maximize: 是否最大化目标，True为最大化，False为最小化
            random_state: 随机种子
        """
        self.model = model
        self.feature_ranges = feature_ranges
        self.acquisition_function = acquisition_function.lower()
        self.maximize = maximize
        self.random_state = random_state
        self.gp_model = None
        
        # 验证采样函数
        valid_acq_funcs = ['ei', 'ucb', 'pi']
        if self.acquisition_function not in valid_acq_funcs:
            raise ValueError(f"不支持的采样函数: {acquisition_function}，可选: {', '.join(valid_acq_funcs)}")
    
    def _generate_grid(self):
        """
        生成特征空间网格
        
        返回:
            特征网格点的DataFrame
        """
        if not self.feature_ranges:
            raise ValueError("未设置特征范围")
        
        grid_points = []
        feature_names = []
        
        # 为每个特征生成网格点
        for feature, range_info in self.feature_ranges.items():
            feature_names.append(feature)
            min_val = range_info['min']
            max_val = range_info['max']
            
            if 'step' in range_info and range_info['step'] > 0:
                # 使用指定的步长
                step = range_info['step']
                points = np.arange(min_val, max_val + step/2, step)  # 加step/2确保包含max_val
            else:
                # 默认使用10个点
                points = np.linspace(min_val, max_val, 10)
            
            grid_points.append(points)
        
        # 生成网格
        mesh = np.meshgrid(*grid_points)
        grid = np.column_stack([m.flatten() for m in mesh])
        
        # 转换为DataFrame
        grid_df = pd.DataFrame(grid, columns=feature_names)
        
        return grid_df
    
    def _fit_gp(self, X, y):
        """
        拟合高斯过程模型
        
        参数:
            X: 特征矩阵
            y: 目标向量
        """
        # 如果需要最小化，则取负值
        y_opt = y if self.maximize else -y
        
        # 创建高斯过程回归模型
        kernel = ConstantKernel() * Matern(nu=2.5)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state
        )
        
        # 拟合模型
        self.gp_model.fit(X, y_opt)
    
    def _expected_improvement(self, X, X_sample, y_sample, xi=0.01):
        """
        计算期望改进(EI)采样函数
        
        参数:
            X: 待评估的点
            X_sample: 已采样的点
            y_sample: 已采样点的目标值
            xi: 探索参数
            
        返回:
            期望改进值
        """
        # 获取当前最优值
        y_best = y_sample.max()
        
        # 预测均值和标准差
        mu, sigma = self.gp_model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        # 计算改进量
        with np.errstate(divide='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _upper_confidence_bound(self, X, X_sample, y_sample, kappa=2.0):
        """
        计算置信上界(UCB)采样函数
        
        参数:
            X: 待评估的点
            X_sample: 已采样的点
            y_sample: 已采样点的目标值
            kappa: 探索参数
            
        返回:
            UCB值
        """
        # 预测均值和标准差
        mu, sigma = self.gp_model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        # 计算UCB
        ucb = mu + kappa * sigma
        
        return ucb
    
    def _probability_improvement(self, X, X_sample, y_sample, xi=0.01):
        """
        计算改进概率(PI)采样函数
        
        参数:
            X: 待评估的点
            X_sample: 已采样的点
            y_sample: 已采样点的目标值
            xi: 探索参数
            
        返回:
            改进概率值
        """
        # 获取当前最优值
        y_best = y_sample.max()
        
        # 预测均值和标准差
        mu, sigma = self.gp_model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        # 计算改进概率
        with np.errstate(divide='warn'):
            Z = (mu - y_best - xi) / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0
        
        return pi
    
    def _calculate_acquisition(self, X, X_sample, y_sample):
        """
        计算采样函数值
        
        参数:
            X: 待评估的点
            X_sample: 已采样的点
            y_sample: 已采样点的目标值
            
        返回:
            采样函数值
        """
        if self.acquisition_function == 'ei':
            return self._expected_improvement(X, X_sample, y_sample)
        elif self.acquisition_function == 'ucb':
            return self._upper_confidence_bound(X, X_sample, y_sample)
        elif self.acquisition_function == 'pi':
            return self._probability_improvement(X, X_sample, y_sample)
        else:
            raise ValueError(f"不支持的采样函数: {self.acquisition_function}")
    
    def recommend_experiments(self, X, y, n_recommendations=3):
        """
        推荐下一轮实验条件
        
        参数:
            X: 已有实验的特征矩阵
            y: 已有实验的目标向量
            n_recommendations: 推荐实验数量
            
        返回:
            推荐实验条件的DataFrame
        """
        # 生成特征空间网格
        grid_df = self._generate_grid()
        
        # 如果使用的是高斯过程模型，直接使用
        if hasattr(self.model, 'kernel_') and 'GaussianProcessRegressor' in str(type(self.model)):
            self.gp_model = self.model
            # 如果需要最小化，则取负值
            y_opt = y if self.maximize else -y
            # 重新拟合模型
            self.gp_model.fit(X, y_opt)
        else:
            # 否则，拟合新的高斯过程模型
            self._fit_gp(X, y)
        
        # 计算采样函数值
        y_opt = y if self.maximize else -y
        acq_values = self._calculate_acquisition(grid_df.values, X.values, y_opt.values)
        
        # 添加采样函数值到网格
        grid_df['acquisition_value'] = acq_values
        
        # 使用原始模型预测目标值
        if self.model is not None:
            grid_df['predicted_value'] = self.model.predict(grid_df.drop('acquisition_value', axis=1))
        
        # 根据采样函数值排序并选择前n个点
        sorted_grid = grid_df.sort_values('acquisition_value', ascending=False)
        recommendations = sorted_grid.head(n_recommendations).reset_index(drop=True)