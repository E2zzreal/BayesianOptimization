import numpy as np
import pandas as pd
from scipy.stats import norm
import random
from abc import ABC, abstractmethod

class SearchStrategy(ABC):
    """
    搜索策略的抽象基类，定义了所有搜索策略必须实现的接口
    """
    
    def __init__(self, feature_ranges, random_state=42):
        """
        初始化搜索策略
        
        参数:
            feature_ranges: 特征范围字典，格式为 {feature_name: {'min': min_val, 'max': max_val, 'step': step_val}}
            random_state: 随机种子
        """
        self.feature_ranges = feature_ranges
        self.random_state = random_state
        self.feature_names = list(feature_ranges.keys())
        np.random.seed(random_state)
        random.seed(random_state)
    
    @abstractmethod
    def search(self, model, acquisition_func, n_points=100, maximize=True):
        """
        执行搜索并返回最优点
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估点的价值
            n_points: 要评估的点的数量
            maximize: 是否最大化目标，True为最大化，False为最小化
            
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
    
    def _random_sample(self, n_points):
        """
        在特征空间中随机采样点
        
        参数:
            n_points: 要采样的点的数量
            
        返回:
            采样点的DataFrame
        """
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
    
    def search(self, model, acquisition_func, n_points=100, maximize=True):
        """
        执行随机搜索
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估点的价值
            n_points: 要评估的点的数量
            maximize: 是否最大化目标，True为最大化，False为最小化
            
        返回:
            搜索结果的DataFrame
        """
        # 随机采样n_points个点
        random_points = self._random_sample(n_points)
        
        # 使用模型预测目标值
        random_points['predicted_value'] = model.predict(random_points)
        
        # 计算采集函数值
        random_points['acquisition_value'] = acquisition_func(random_points.values)
        
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
    
    def search(self, model, acquisition_func, n_points=None, maximize=True):
        """
        执行网格搜索
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估点的价值
            n_points: 不使用，网格搜索使用步长确定点数
            maximize: 是否最大化目标，True为最大化，False为最小化
            
        返回:
            搜索结果的DataFrame
        """
        grid_df = self._generate_grid()
        
        # 使用模型预测目标值
        grid_df['predicted_value'] = model.predict(grid_df)
        
        # 计算采集函数值
        grid_df['acquisition_value'] = acquisition_func(grid_df.values)
        
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
        mesh = np.meshgrid(*grid_points, indexing='ij')  # 使用'ij'索引确保维度顺序一致
        grid = np.column_stack([m.flatten() for m in mesh])
        
        # 转换为DataFrame
        grid_df = pd.DataFrame(grid, columns=self.feature_names)
        
        return grid_df


class GeneticAlgorithm(SearchStrategy):
    """
    遗传算法搜索策略
    """
    
    def __init__(self, feature_ranges, random_state=42, population_size=50, n_generations=10, 
                 crossover_prob=0.8, mutation_prob=0.2):
        """
        初始化遗传算法
        
        参数:
            feature_ranges: 特征范围字典
            random_state: 随机种子
            population_size: 种群大小
            n_generations: 迭代代数
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
        """
        super().__init__(feature_ranges, random_state)
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
    
    def search(self, model, acquisition_func, n_points=100, maximize=True):
        """
        执行遗传算法搜索
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估个体的适应度
            n_points: 返回的最优点数量
            maximize: 是否最大化目标
            
        返回:
            搜索结果的DataFrame
        """
        # 初始化种群
        population = self._random_sample(self.population_size)
        
        # 迭代进化
        for generation in range(self.n_generations):
            # 计算适应度
            population['predicted_value'] = model.predict(population)
            population['fitness'] = acquisition_func(population.values)
            
            # 选择父代
            parents = self._selection(population)
            
            # 生成子代
            offspring = self._crossover(parents)
            offspring = self._mutation(offspring)
            
            # 更新种群
            population = offspring
        
        # 计算最终种群的适应度
        population['predicted_value'] = model.predict(population)
        population['acquisition_value'] = acquisition_func(population.values)
        
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
        变异操作
        
        参数:
            offspring: 子代种群
            
        返回:
            变异后的子代
        """
        for i in range(self.population_size):
            for j, feature in enumerate(self.feature_names):
                # 以变异概率决定是否进行变异
                if np.random.random() < self.mutation_prob:
                    # 获取特征范围
                    range_info = self.feature_ranges[feature]
                    min_val = range_info['min']
                    max_val = range_info['max']
                    
                    # 生成新的特征值
                    offspring.loc[i, feature] = np.random.uniform(min_val, max_val)
        
        return offspring


class ParticleSwarmOptimization(SearchStrategy):
    """
    粒子群优化搜索策略
    """
    
    def __init__(self, feature_ranges, random_state=42, n_particles=30, n_iterations=20, 
                 inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
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
        """
        super().__init__(feature_ranges, random_state)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
    
    def search(self, model, acquisition_func, n_points=100, maximize=True):
        """
        执行粒子群优化搜索
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估粒子的适应度
            n_points: 返回的最优点数量
            maximize: 是否最大化目标
            
        返回:
            搜索结果的DataFrame
        """
        # 初始化粒子位置和速度
        particles = self._random_sample(self.n_particles)
        velocities = self._initialize_velocities()
        
        # 初始化个体最优位置和适应度
        particles['predicted_value'] = model.predict(particles)
        particles['fitness'] = acquisition_func(particles.values)
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
            
            # 计算新位置的适应度
            particles['predicted_value'] = model.predict(particles)
            particles['fitness'] = acquisition_func(particles.values)
            
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
                 initial_temp=100, cooling_rate=0.95, n_neighbors=5):
        """
        初始化模拟退火
        
        参数:
            feature_ranges: 特征范围字典
            random_state: 随机种子
            n_iterations: 迭代次数
            initial_temp: 初始温度
            cooling_rate: 冷却率
            n_neighbors: 每次迭代生成的邻居数量
        """
        super().__init__(feature_ranges, random_state)
        self.n_iterations = n_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.n_neighbors = n_neighbors
    
    def search(self, model, acquisition_func, n_points=100, maximize=True):
        """
        执行模拟退火搜索
        
        参数:
            model: 训练好的模型，用于预测
            acquisition_func: 采集函数，用于评估解的质量
            n_points: 返回的最优点数量
            maximize: 是否最大化目标
            
        返回:
            搜索结果的DataFrame
        """
        # 初始化当前解
        current_solution = self._random_sample(1)
        current_solution['predicted_value'] = model.predict(current_solution)
        current_solution['acquisition_value'] = acquisition_func(current_solution.values)
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
            # 生成邻居解
            neighbors = self._generate_neighbors(current_solution)
            neighbors['predicted_value'] = model.predict(neighbors)
            neighbors['acquisition_value'] = acquisition_func(neighbors.values)
            
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