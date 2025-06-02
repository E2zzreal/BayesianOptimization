import numpy as np
import pandas as pd
import random
from abc import ABC, abstractmethod


def apply_sum_constraint(points, max_sum=None):
    """
    应用特征和约束，过滤掉不满足约束的点
    
    参数:
        points: 候选点矩阵(DataFrame或numpy array)
        max_sum: 最大特征和阈值，如果为None则不应用约束
        
    返回:
        满足约束的候选点(与输入类型相同)
    """
    if max_sum is None:
        return points
    
    # 处理DataFrame输入
    if isinstance(points, pd.DataFrame):
        # 计算每行特征的和
        row_sums = points.sum(axis=1)
        # 找到满足约束的行
        valid_mask = row_sums <= max_sum
        return points[valid_mask].reset_index(drop=True)
    
    # 处理numpy array输入
    elif isinstance(points, np.ndarray):
        # 计算每行特征的和
        row_sums = np.sum(points, axis=1)
        # 找到满足约束的行
        valid_mask = row_sums <= max_sum
        return points[valid_mask]
    
    else:
        raise TypeError("points必须是pandas DataFrame或numpy array")


def generate_constrained_samples(feature_ranges, n_points, max_sum=None, max_attempts=10000):
    """
    生成满足约束的随机样本点
    
    参数:
        feature_ranges: 特征范围字典
        n_points: 需要生成的点数
        max_sum: 最大特征和约束
        max_attempts: 最大尝试次数
        
    返回:
        满足约束的样本点DataFrame
    """
    if max_sum is None:
        # 如果没有约束，直接随机采样
        samples = {}
        feature_names = list(feature_ranges.keys())
        
        for feature in feature_names:
            range_info = feature_ranges[feature]
            min_val = range_info['min']
            max_val = range_info['max']
            samples[feature] = np.random.uniform(min_val, max_val, n_points)
        
        return pd.DataFrame(samples)
    
    # 有约束的情况，使用拒绝采样
    feature_names = list(feature_ranges.keys())
    valid_samples = []
    attempts = 0
    
    while len(valid_samples) < n_points and attempts < max_attempts:
        # 生成一批候选样本
        batch_size = min(n_points * 2, 1000)  # 批量生成以提高效率
        batch_samples = {}
        
        for feature in feature_names:
            range_info = feature_ranges[feature]
            min_val = range_info['min']
            max_val = range_info['max']
            batch_samples[feature] = np.random.uniform(min_val, max_val, batch_size)
        
        batch_df = pd.DataFrame(batch_samples)
        
        # 应用约束过滤
        constrained_batch = apply_sum_constraint(batch_df, max_sum)
        
        # 添加到有效样本列表
        if len(constrained_batch) > 0:
            valid_samples.append(constrained_batch)
        
        attempts += batch_size
    
    if not valid_samples:
        raise ValueError(f"在{max_attempts}次尝试后无法生成满足约束(sum <= {max_sum})的样本点。请检查约束设置是否合理。")
    
    # 合并所有有效样本
    all_valid = pd.concat(valid_samples, ignore_index=True)
    
    # 如果生成的样本超过需要的数量，随机选择
    if len(all_valid) > n_points:
        return all_valid.sample(n=n_points, random_state=42).reset_index(drop=True)
    elif len(all_valid) < n_points:
        print(f"警告: 只能生成{len(all_valid)}个满足约束的样本点，少于请求的{n_points}个")
    
    return all_valid

class SearchStrategy(ABC):
    """
    搜索策略的抽象基类，定义了所有搜索策略必须实现的接口
    """
    
    def __init__(self, feature_ranges, random_state=42, max_sum=None):
        """
        初始化搜索策略
        
        参数:
            feature_ranges: 特征范围字典，格式为 {feature_name: {'min': min_val, 'max': max_val, 'step': step_val}}
            random_state: 随机种子
            max_sum: 特征和的最大值约束，如果为None则不应用约束
        """
        if not isinstance(feature_ranges, dict):
            raise TypeError("feature_ranges 必须是字典类型")
        self.feature_ranges = feature_ranges
        self.random_state = random_state
        self.max_sum = max_sum
        self.feature_names = list(feature_ranges.keys())
        if len(self.feature_names) != len(feature_ranges):
             # This should theoretically not happen if feature_ranges is a dict
             raise ValueError("特征名称列表长度与特征范围字典长度不匹配。")
        np.random.seed(random_state)
        random.seed(random_state)
    
    @abstractmethod
    def search(self, model, acquisition_func, n_points=100, maximize=True, max_sum=None):
        """
        执行搜索并返回最优点
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估点的价值
            n_points: 要评估的点的数量
            maximize: 是否最大化目标，True为最大化，False为最小化
            max_sum: 特征和的最大值约束，如果为None则使用初始化时的值
            
        返回:
            搜索结果的DataFrame
        """
        pass
    
    def calculate_memory_usage(self, n_points):
        """
        估算内存使用量
        
        参数:
            n_points: 要评估的点的数量
            
        返回:
            tuple: (预估内存消耗(MB), 警告信息)
        """
        # 估算数据存储内存消耗 (每个浮点数8字节，再加上DataFrame的开销)
        # 假设每个点有len(feature_ranges)个特征，加上2个额外列(预测值和采样函数值)
        num_columns = len(self.feature_ranges) + 2
        data_memory_bytes = n_points * num_columns * 8  # 8字节/浮点数
        
        # 算法特定的内存开销 (由子类实现)
        method_memory_bytes = self._calculate_method_memory(n_points)
        
        # 总内存消耗
        total_memory_bytes = data_memory_bytes + method_memory_bytes
        total_memory_mb = total_memory_bytes / (1024 * 1024)  # 转换为MB
        
        # 生成警告信息
        warning = ""
        if total_memory_mb > 8000:  # 超过8GB内存
            warning = "警告: 预估内存消耗超过8GB，可能导致系统不稳定或崩溃。建议减少评估点数量。"
        elif total_memory_mb > 4000:  # 超过4GB内存
            warning = "警告: 预估内存消耗较大(>4GB)，可能影响系统性能。"
        elif total_memory_mb > 2000:  # 超过2GB内存
            warning = "提示: 预估内存消耗中等(>2GB)，在低配置机器上可能影响性能。"
        
        return total_memory_mb, warning
    
    @abstractmethod
    def _calculate_method_memory(self, n_points):
        """
        计算算法特定的内存开销
        
        参数:
            n_points: 要评估的点的数量
            
        返回:
            算法特定的内存开销(字节)
        """
        pass
    
    def _random_sample(self, n_points, max_sum=None):
        """
        在特征空间中随机采样点
        
        参数:
            n_points: 要采样的点的数量
            max_sum: 特征和的最大值约束，如果为None则使用self.max_sum
            
        返回:
            采样点的DataFrame
        """
        # 确定使用的约束值
        constraint = max_sum if max_sum is not None else self.max_sum
        
        # 如果有约束，使用约束采样函数
        if constraint is not None:
            return generate_constrained_samples(self.feature_ranges, n_points, constraint)
        
        # 无约束的情况，使用原始采样方法
        samples = {}
        
        for feature in self.feature_names:
            range_info = self.feature_ranges[feature]
            min_val = range_info['min']
            max_val = range_info['max']
            
            # 随机采样
            samples[feature] = np.random.uniform(min_val, max_val, n_points)
        
        return pd.DataFrame(samples)


class RandomSearch(SearchStrategy):
    """
    随机搜索策略
    """
    
    def search(self, model, acquisition_func, n_points=100, maximize=True, max_sum=None):
        """
        执行随机搜索
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估点的价值
            n_points: 要评估的点的数量
            maximize: 是否最大化目标，True为最大化，False为最小化
            max_sum: 特征和的最大值约束，如果为None则使用初始化时的值
            
        返回:
            搜索结果的DataFrame
        """
        # 确定使用的约束值
        constraint = max_sum if max_sum is not None else self.max_sum
        
        # 随机采样n_points个点（考虑约束）
        random_points = self._random_sample(n_points, constraint)
        
        # 如果由于约束导致生成的点数不足，发出警告
        if len(random_points) < n_points:
            print(f"警告: 由于约束限制，只生成了{len(random_points)}个点，少于请求的{n_points}个")
        
        # 使用模型预测目标值
        random_points['predicted_value'] = model.predict(random_points)
        
        # 计算采集函数值 (只传递特征列)
        random_points['acquisition_value'] = acquisition_func(random_points[self.feature_names].values)
        
        # 根据采集函数值排序
        sorted_points = random_points.sort_values('acquisition_value', ascending=False)
        
        return sorted_points
    
    def _calculate_method_memory(self, n_points):
        """
        计算随机搜索的内存开销
        
        参数:
            n_points: 要评估的点的数量
            
        返回:
            随机搜索的内存开销(字节)
        """
        # 随机搜索的额外内存开销主要是存储采样点
        return n_points * len(self.feature_ranges) * 8

class GridSearch(SearchStrategy):
    """
    网格搜索策略
    """
    
    def search(self, model, acquisition_func, n_points=None, maximize=True, max_sum=None):
        """
        执行网格搜索
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估点的价值
            n_points: 不使用，网格搜索使用步长确定点数
            maximize: 是否最大化目标，True为最大化，False为最小化
            max_sum: 特征和的最大值约束，如果为None则使用初始化时的值
            
        返回:
            搜索结果的DataFrame
        """
        # 确定使用的约束值
        constraint = max_sum if max_sum is not None else self.max_sum
        
        grid_df = self._generate_grid()
        
        # 应用约束过滤
        if constraint is not None:
            original_size = len(grid_df)
            grid_df = apply_sum_constraint(grid_df, constraint)
            if len(grid_df) < original_size:
                print(f"约束过滤: 从{original_size}个网格点中保留了{len(grid_df)}个满足约束的点")
        
        # 如果过滤后没有点，抛出错误
        if len(grid_df) == 0:
            raise ValueError(f"应用约束(sum <= {constraint})后，没有网格点满足条件。请调整约束值或特征范围。")
        
        # 使用模型预测目标值
        grid_df['predicted_value'] = model.predict(grid_df)
        
        # 计算采集函数值 (只传递特征列)
        grid_df['acquisition_value'] = acquisition_func(grid_df[self.feature_names].values)
        
        # 根据采集函数值排序
        sorted_grid = grid_df.sort_values('acquisition_value', ascending=False)
        
        return sorted_grid
    
    def _calculate_method_memory(self, n_points):
        """
        计算网格搜索的内存开销
        
        参数:
            n_points: 不使用，网格搜索使用步长确定点数
            
        返回:
            网格搜索的内存开销(字节)
        """
        # 计算每个特征的点数
        points_per_feature = []
        for feature, range_info in self.feature_ranges.items():
            min_val = range_info['min']
            max_val = range_info['max']
            
            if 'step' in range_info and range_info['step'] > 0:
                # 使用指定的步长
                step = range_info['step']
                num_points = int(np.ceil((max_val - min_val) / step)) + 1
            else:
                # 默认使用10个点
                num_points = 10
            
            points_per_feature.append(num_points)
        
        # 计算总点数
        total_points = np.prod(points_per_feature)
        
        # 网格搜索的额外内存开销
        # 主要是存储网格点和中间计算结果
        grid_memory_bytes = total_points * len(self.feature_ranges) * 8  # 网格点
        intermediate_memory_bytes = total_points * 3 * 8  # 中间计算结果
        
        return grid_memory_bytes + intermediate_memory_bytes
    
    def _generate_neighbors(self, current_solution):
        """
        生成邻居解
        
        参数:
            current_solution: 当前解
            
        返回:
            邻居解的DataFrame
        """
        neighbors = []
        current_values = current_solution.values[0]
        
        for _ in range(self.n_neighbors):
            neighbor = {}
            for i, (feature, range_info) in enumerate(self.feature_ranges.items()):
                min_val = range_info['min']
                max_val = range_info['max']
                current_val = current_values[i]
                
                # 在当前值附近随机扰动，扰动幅度随温度降低而减小
                perturbation = np.random.normal(0, (max_val - min_val) * 0.1 * self.current_temp / self.initial_temp)
                new_val = current_val + perturbation
                
                # 确保新值在特征范围内
                new_val = max(min_val, min(max_val, new_val))
                neighbor[feature] = new_val
            
            neighbors.append(neighbor)
        
        return pd.DataFrame(neighbors)
    
    def _generate_grid(self):
        """
        生成特征空间网格
        
        返回:
            特征网格点的DataFrame
        """
        if not self.feature_ranges:
            raise ValueError("未设置特征范围")
        
        grid_points = []
        
        # 为每个特征生成网格点
        for feature in self.feature_names:
            range_info = self.feature_ranges[feature]
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
        grid = np.vstack([m.ravel() for m in mesh]).T
        
        # 转换为DataFrame
        grid_df = pd.DataFrame(grid, columns=self.feature_names)
        
        return grid_df


class GeneticAlgorithm(SearchStrategy):
    """
    遗传算法搜索策略
    """
    
    def __init__(self, feature_ranges, random_state=42, population_size=50, n_generations=10,
                 crossover_prob=0.8, mutation_prob=0.2, max_sum=None):
        """
        初始化遗传算法
        
        参数:
            feature_ranges: 特征范围字典
            random_state: 随机种子
            population_size: 种群大小
            n_generations: 迭代代数
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            max_sum: 特征和的最大值约束
        """
        super().__init__(feature_ranges, random_state, max_sum)
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
    
    def search(self, model, acquisition_func, n_points=100, maximize=True, max_sum=None):
        """
        执行遗传算法搜索
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估个体的适应度
            n_points: 返回的最优点数量
            maximize: 是否最大化目标
            max_sum: 特征和的最大值约束，如果为None则使用初始化时的值
            
        返回:
            搜索结果的DataFrame
        """
        # 确定使用的约束值
        constraint = max_sum if max_sum is not None else self.max_sum
        
        # 初始化种群（考虑约束）
        population = self._random_sample(self.population_size, constraint)
        
        # 如果由于约束导致种群大小不足，发出警告
        if len(population) < self.population_size:
            print(f"警告: 由于约束限制，种群大小为{len(population)}，少于设定的{self.population_size}")
            self.population_size = len(population)  # 调整种群大小
        
        # 迭代进化
        for generation in range(self.n_generations):
            # 计算适应度 (只传递特征列)
            population['predicted_value'] = model.predict(population[self.feature_names]) # 预测也应只用特征列
            population['fitness'] = acquisition_func(population[self.feature_names].values)
            
            # 选择父代
            parents = self._selection(population)
            
            # 生成子代
            offspring = self._crossover(parents)
            offspring = self._mutation(offspring)
            
            # 更新种群
            population = offspring
        
        # 计算最终种群的适应度 (只传递特征列)
        population['predicted_value'] = model.predict(population[self.feature_names]) # 预测也应只用特征列
        population['acquisition_value'] = acquisition_func(population[self.feature_names].values)
        
        # 根据适应度排序
        sorted_population = population.sort_values('acquisition_value', ascending=False)
        
        # 返回前n_points个最优个体
        return sorted_population.head(n_points).reset_index(drop=True)
    
    def _calculate_method_memory(self, n_points):
        """
        计算遗传算法的内存开销
        
        参数:
            n_points: 评估点数量
            
        返回:
            遗传算法的内存开销(字节)
        """
        # 种群内存
        population_memory = self.population_size * len(self.feature_ranges) * 8
        
        # 适应度计算内存
        fitness_memory = self.population_size * 2 * 8  # 预测值和适应度
        
        # 选择、交叉和变异操作的临时内存
        operation_memory = self.population_size * len(self.feature_ranges) * 8 * 3
        
        return population_memory + fitness_memory + operation_memory
    
    def _selection(self, population):
        """
        选择操作 - 使用轮盘赌选择
        
        参数:
            population: 当前种群
            
        返回:
            选择的父代
        """
        # 提取适应度
        fitness = population['fitness'].values
        
        # 确保适应度为正
        if np.min(fitness) < 0:
            fitness = fitness - np.min(fitness) + 1e-6
        
        # 计算选择概率
        prob = fitness / np.sum(fitness)
        
        # 选择父代
        selected_indices = np.random.choice(
            population.index, 
            size=self.population_size, 
            replace=True, 
            p=prob
        )
        
        # 返回选择的父代
        return population.loc[selected_indices, self.feature_names].reset_index(drop=True)
    
    def _crossover(self, parents):
        """
        交叉操作
        
        参数:
            parents: 父代种群
            
        返回:
            交叉后的子代
        """
        offspring = parents.copy()
        
        for i in range(0, self.population_size, 2):
            # 如果达到种群大小的奇数位置，跳过
            if i + 1 >= self.population_size:
                break
                
            # 以交叉概率决定是否进行交叉
            if np.random.random() < self.crossover_prob:
                # 随机选择交叉点
                crossover_point = np.random.randint(1, len(self.feature_names))
                
                # 交换特征
                for j in range(crossover_point, len(self.feature_names)):
                    feature = self.feature_names[j]
                    offspring.loc[i, feature], offspring.loc[i+1, feature] = \
                        offspring.loc[i+1, feature], offspring.loc[i, feature]
        
        return offspring
    
    def _mutation(self, offspring):
        """
        变异操作（支持约束）
        
        参数:
            offspring: 子代种群
            
        返回:
            变异后的子代
        """
        for i in range(len(offspring)):  # 使用实际种群大小
            for j, feature in enumerate(self.feature_names):
                # 以变异概率决定是否进行变异
                if np.random.random() < self.mutation_prob:
                    # 如果有约束，需要确保变异后仍满足约束
                    if self.max_sum is not None:
                        # 计算其他特征的和
                        other_features_sum = offspring.loc[i, [f for f in self.feature_names if f != feature]].sum()
                        
                        # 计算当前特征的可用范围
                        range_info = self.feature_ranges[feature]
                        min_val = range_info['min']
                        max_val = min(range_info['max'], self.max_sum - other_features_sum)
                        
                        # 如果可用范围有效，则进行变异
                        if max_val > min_val:
                            offspring.loc[i, feature] = np.random.uniform(min_val, max_val)
                    else:
                        # 无约束的情况，正常变异
                        range_info = self.feature_ranges[feature]
                        min_val = range_info['min']
                        max_val = range_info['max']
                        offspring.loc[i, feature] = np.random.uniform(min_val, max_val)
        
        # 如果有约束，过滤掉不满足约束的个体
        if self.max_sum is not None:
            offspring = apply_sum_constraint(offspring, self.max_sum)
        
        return offspring


class ParticleSwarmOptimization(SearchStrategy):
    """
    粒子群优化搜索策略
    """
    
    def __init__(self, feature_ranges, random_state=42, n_particles=30, n_iterations=20,
                 inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5, max_sum=None):
        """
        初始化粒子群优化
        
        参数:
            feature_ranges: 特征范围字典
            random_state: 随机种子
            n_particles: 粒子数量
            n_iterations: 迭代次数
            inertia_weight: 惯性权重
            cognitive_weight: 认知权重
            social_weight: 社会权重
            max_sum: 特征和的最大值约束
        """
        super().__init__(feature_ranges, random_state, max_sum)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
    
    def search(self, model, acquisition_func, n_points=100, maximize=True, max_sum=None):
        """
        执行粒子群优化搜索
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估粒子的适应度
            n_points: 返回的最优点数量
            maximize: 是否最大化目标
            max_sum: 特征和的最大值约束，如果为None则使用初始化时的值
            
        返回:
            搜索结果的DataFrame
        """
        # 确定使用的约束值
        constraint = max_sum if max_sum is not None else self.max_sum
        
        # 初始化粒子位置和速度（考虑约束）
        particles = self._random_sample(self.n_particles, constraint)
        
        # 如果由于约束导致粒子数量不足，发出警告并调整
        if len(particles) < self.n_particles:
            print(f"警告: 由于约束限制，粒子数量为{len(particles)}，少于设定的{self.n_particles}")
            self.n_particles = len(particles)
        
        velocities = self._initialize_velocities()
        
        # 初始化个体最优位置和适应度 (只传递特征列)
        particles['predicted_value'] = model.predict(particles[self.feature_names]) # 预测也应只用特征列
        particles['fitness'] = acquisition_func(particles[self.feature_names].values)
        pbest_positions = particles[self.feature_names].copy()
        pbest_fitness = particles['fitness'].copy()
        
        # 初始化全局最优位置和适应度
        gbest_idx = pbest_fitness.idxmax()
        gbest_position = pbest_positions.loc[gbest_idx].copy()
        gbest_fitness = pbest_fitness.loc[gbest_idx]
        
        # 存储所有评估过的点
        all_positions = [particles[self.feature_names].copy()]
        all_fitness = [particles['fitness'].copy()]
        
        # 迭代优化
        for iteration in range(self.n_iterations):
            # 更新粒子速度和位置
            for i in range(self.n_particles):
                # 更新速度
                for j, feature in enumerate(self.feature_names):
                    r1, r2 = np.random.random(2)
                    cognitive_component = self.cognitive_weight * r1 * (pbest_positions.loc[i, feature] - particles.loc[i, feature])
                    social_component = self.social_weight * r2 * (gbest_position[feature] - particles.loc[i, feature])
                    velocities.loc[i, feature] = self.inertia_weight * velocities.loc[i, feature] + cognitive_component + social_component
                
                # 更新位置
                for feature in self.feature_names:
                    particles.loc[i, feature] += velocities.loc[i, feature]
                    
                    # 边界处理
                    range_info = self.feature_ranges[feature]
                    min_val = range_info['min']
                    max_val = range_info['max']
                    
                    if particles.loc[i, feature] < min_val:
                        particles.loc[i, feature] = min_val
                        velocities.loc[i, feature] *= -0.5  # 反弹
                    elif particles.loc[i, feature] > max_val:
                        particles.loc[i, feature] = max_val
                        velocities.loc[i, feature] *= -0.5  # 反弹
            
            # 计算新位置的适应度 (只传递特征列)
            particles['predicted_value'] = model.predict(particles[self.feature_names]) # 预测也应只用特征列
            particles['fitness'] = acquisition_func(particles[self.feature_names].values)
            
            # 更新个体最优
            for i in range(self.n_particles):
                if particles.loc[i, 'fitness'] > pbest_fitness.loc[i]:
                    pbest_positions.loc[i] = particles.loc[i, self.feature_names]
                    pbest_fitness.loc[i] = particles.loc[i, 'fitness']
            
            # 更新全局最优
            current_best_idx = pbest_fitness.idxmax()
            if pbest_fitness.loc[current_best_idx] > gbest_fitness:
                gbest_position = pbest_positions.loc[current_best_idx].copy()
                gbest_fitness = pbest_fitness.loc[current_best_idx]
            
            # 存储当前迭代的位置和适应度
            all_positions.append(particles[self.feature_names].copy())
            all_fitness.append(particles['fitness'].copy())
        
        # 合并所有评估过的点
        all_particles = pd.concat(all_positions).reset_index(drop=True)
        all_particles['fitness'] = pd.concat(all_fitness).reset_index(drop=True)
        
        # 去除重复点
        all_particles = all_particles.drop_duplicates(subset=self.feature_names).reset_index(drop=True)
        
        # 计算最终结果的预测值和采集函数值
        all_particles['predicted_value'] = model.predict(all_particles[self.feature_names])
        all_particles['acquisition_value'] = all_particles['fitness']
        all_particles = all_particles.drop(columns=['fitness'])
        
        # 根据采集函数值排序
        sorted_particles = all_particles.sort_values('acquisition_value', ascending=False)
        
        # 返回前n_points个最优粒子
        return sorted_particles.head(n_points).reset_index(drop=True)
    
    def _calculate_method_memory(self, n_points):
        """
        计算粒子群优化的内存开销
        
        参数:
            n_points: 评估点数量
            
        返回:
            粒子群优化的内存开销(字节)
        """
        # 粒子位置内存
        particles_memory = self.n_particles * len(self.feature_ranges) * 8
        
        # 粒子速度内存
        velocities_memory = self.n_particles * len(self.feature_ranges) * 8
        
        # 个体最优位置和适应度内存
        pbest_memory = self.n_particles * (len(self.feature_ranges) + 1) * 8
        
        # 全局最优位置和适应度内存
        gbest_memory = (len(self.feature_ranges) + 1) * 8
        
        # 迭代过程中的临时内存
        temp_memory = self.n_particles * len(self.feature_ranges) * 8 * 2
        
        # 所有评估过的点的内存
        all_points_memory = self.n_particles * self.n_iterations * (len(self.feature_ranges) + 1) * 8
        
        return particles_memory + velocities_memory + pbest_memory + gbest_memory + temp_memory + all_points_memory
    
    def _initialize_velocities(self):
        """
        初始化粒子速度
        
        返回:
            初始速度的DataFrame
        """
        velocities = {}
        
        for feature in self.feature_names:
            range_info = self.feature_ranges[feature]
            min_val = range_info['min']
            max_val = range_info['max']
            
            # 初始速度为范围的10%
            velocity_range = (max_val - min_val) * 0.1
            velocities[feature] = np.random.uniform(-velocity_range, velocity_range, self.n_particles)
        
        return pd.DataFrame(velocities)


class SimulatedAnnealing(SearchStrategy):
    """
    模拟退火搜索策略
    """
    
    def __init__(self, feature_ranges, random_state=42, n_iterations=100,
                 initial_temp=100, cooling_rate=0.95, n_neighbors=5, max_sum=None):
        """
        初始化模拟退火
        
        参数:
            feature_ranges: 特征范围字典
            random_state: 随机种子
            n_iterations: 迭代次数
            initial_temp: 初始温度
            cooling_rate: 冷却率
            n_neighbors: 每次迭代生成的邻居数量
            max_sum: 特征和的最大值约束
        """
        super().__init__(feature_ranges, random_state, max_sum)
        self.n_iterations = n_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.n_neighbors = n_neighbors
    
    def search(self, model, acquisition_func, n_points=100, maximize=True, max_sum=None):
        """
        执行模拟退火搜索
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估解的质量
            n_points: 返回的最优点数量
            maximize: 是否最大化目标
            max_sum: 特征和的最大值约束，如果为None则使用初始化时的值
            
        返回:
            搜索结果的DataFrame
        """
        # 确定使用的约束值
        constraint = max_sum if max_sum is not None else self.max_sum
        
        # 初始化当前解（考虑约束）
        current_solution = self._random_sample(1, constraint)
        
        # 如果由于约束无法生成初始解，抛出错误
        if len(current_solution) == 0:
            raise ValueError(f"无法生成满足约束(sum <= {constraint})的初始解。请调整约束值或特征范围。")
        current_solution['predicted_value'] = model.predict(current_solution[self.feature_names]) # 预测也应只用特征列
        current_solution['acquisition_value'] = acquisition_func(current_solution[self.feature_names].values)
        current_fitness = current_solution['acquisition_value'].values[0]
        
        # 初始化最优解
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        # 存储所有评估过的点
        all_solutions = [current_solution.copy()]
        
        # 初始化温度
        temp = self.initial_temp
        self.current_temp = temp  # 保存当前温度用于邻居生成
        
        # 迭代优化
        for iteration in range(self.n_iterations):
            # 生成邻居解 (只传递特征列)
            neighbors = self._generate_neighbors(current_solution)
            neighbors['predicted_value'] = model.predict(neighbors[self.feature_names]) # 预测也应只用特征列
            neighbors['acquisition_value'] = acquisition_func(neighbors[self.feature_names].values)
            
            # 选择最佳邻居
            best_neighbor_idx = neighbors['acquisition_value'].idxmax()
            best_neighbor = neighbors.loc[best_neighbor_idx].copy()
            best_neighbor_fitness = best_neighbor['acquisition_value']
            
            # 决定是否接受新解
            if best_neighbor_fitness > current_fitness:  # 如果新解更好，直接接受
                current_solution = pd.DataFrame([best_neighbor])
                current_fitness = best_neighbor_fitness
                
                # 更新全局最优解
                if best_neighbor_fitness > best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = best_neighbor_fitness
            else:  # 如果新解更差，以一定概率接受
                # 计算接受概率
                delta = best_neighbor_fitness - current_fitness
                accept_prob = np.exp(delta / temp)
                
                if np.random.random() < accept_prob:
                    current_solution = pd.DataFrame([best_neighbor])
                    current_fitness = best_neighbor_fitness
            
            # 存储当前解
            all_solutions.append(current_solution.copy())
            
            # 降低温度 - 使用自适应冷却策略
            progress = iteration / self.n_iterations
            adaptive_cooling = self.cooling_rate * (1 + 0.5 * np.sin(np.pi * progress))  # 正弦波调整冷却率
            temp *= adaptive_cooling
            self.current_temp = temp  # 更新当前温度
        
        # 合并所有评估过的点
        all_solutions_df = pd.concat(all_solutions).reset_index(drop=True)
        
        # 去除重复点
        all_solutions_df = all_solutions_df.drop_duplicates(subset=self.feature_names).reset_index(drop=True)
        
        # 根据采集函数值排序
        sorted_solutions = all_solutions_df.sort_values('acquisition_value', ascending=False)
        
        # 返回前n_points个最优解
        return sorted_solutions.head(n_points).reset_index(drop=True)
    
    def _calculate_method_memory(self, n_points):
        """
        计算模拟退火的内存开销
        
        参数:
            n_points: 评估点数量
            
        返回:
            模拟退火的内存开销(字节)
        """
        # 当前解内存
        current_solution_memory = len(self.feature_ranges) * 8 * 3  # 特征、预测值和采集函数值
        
        # 最优解内存
        best_solution_memory = len(self.feature_ranges) * 8 * 3
        
        # 邻居解内存
        neighbors_memory = self.n_neighbors * len(self.feature_ranges) * 8 * 3
        
        # 所有评估过的点的内存
        all_solutions_memory = self.n_iterations * len(self.feature_ranges) * 8 * 3
        
        return current_solution_memory + best_solution_memory + neighbors_memory + all_solutions_memory
    
    def _generate_neighbors(self, current_solution):
        """
        生成邻居解（支持约束）
        
        参数:
            current_solution: 当前解
            
        返回:
            邻居解的DataFrame
        """
        neighbors = []
        current_values = current_solution.values[0]
        
        attempts = 0
        max_attempts = self.n_neighbors * 10  # 最大尝试次数
        
        while len(neighbors) < self.n_neighbors and attempts < max_attempts:
            neighbor = {}
            for i, (feature, range_info) in enumerate(self.feature_ranges.items()):
                min_val = range_info['min']
                max_val = range_info['max']
                current_val = current_values[i]
                
                # 在当前值附近随机扰动，扰动幅度随温度降低而减小
                perturbation = np.random.normal(0, (max_val - min_val) * 0.1 * self.current_temp / self.initial_temp)
                new_val = current_val + perturbation
                
                # 确保新值在特征范围内
                new_val = max(min_val, min(max_val, new_val))
                neighbor[feature] = new_val
            
            # 检查约束
            if self.max_sum is None or sum(neighbor.values()) <= self.max_sum:
                neighbors.append(neighbor)
            
            attempts += 1
        
        # 如果生成的邻居数量不足，发出警告
        if len(neighbors) < self.n_neighbors:
            print(f"警告: 由于约束限制，只生成了{len(neighbors)}个邻居解，少于设定的{self.n_neighbors}个")
        
        return pd.DataFrame(neighbors)
