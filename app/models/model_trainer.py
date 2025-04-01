from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from math import sqrt

class ModelTrainer:
    """
    A class for training and evaluating machine learning models.
    """
    
    def __init__(self, models=None, metric='r2', cv_folds=5, test_size=0.2, random_state=42, n_iter=30):
        """
        Initialize the model trainer.
        
        Args:
            models (list): List of model names to train
            metric (str): Evaluation metric ('r2' or 'rmse')
            cv_folds (int): Number of cross-validation folds
            test_size (float): Test set size for train-test split
            random_state (int): Random seed
            n_iter (int): Number of iterations for Bayesian optimization
        """
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        # 确保metric始终为标准格式
        self.metric = 'r2' if metric in ['r2', 'r²'] else metric
        self.best_model = None

        # 定义超参数空间
        from scipy.stats import uniform, loguniform, randint
        
        self.param_spaces = {
            'lasso': {'alpha': loguniform(0.01, 1.0)},
            'random_forest': {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 15),
                'min_samples_split': randint(2, 10)
            },
            'xgboost': {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 10),
                'learning_rate': loguniform(0.01, 0.3)
            },
            'svr': {
                'C': loguniform(0.1, 10.0),
                'epsilon': loguniform(0.01, 0.2)
            },
            'gaussian_process': {
                'alpha': loguniform(1e-11, 1e-3),
                'kernel__k1__constant_value': uniform(0.5, 1.5),  # ConstantKernel参数
                'kernel__k2__length_scale': uniform(0.1, 1.5)  # RBF核参数
            }
        }
        
        # Define available models
        self.available_models = {
            'lasso': Lasso(alpha=0.1, random_state=random_state),
            'random_forest': RandomForestRegressor(random_state=random_state),
            'xgboost': XGBRegressor(random_state=random_state),
            'svr': SVR(kernel='rbf'),
            'gaussian_process': GaussianProcessRegressor(
                kernel=ConstantKernel() * RBF(),
                alpha=1e-6,
                normalize_y=True,
                random_state=random_state
            )
        }
        
        # Select models to train
        if models is None:
            self.models = self.available_models
        else:
            model_map = {
                'Lasso': 'lasso',
                '随机森林': 'random_forest',
                'XGBoost': 'xgboost',
                'SVR': 'svr',
                '高斯过程': 'gaussian_process'
            }
            
            self.models = {}
            for model in models:
                model_key = model_map.get(model, model.lower())
                if model_key in self.available_models:
                    self.models[model_key] = self.available_models[model_key]
    
    def train_model(self, X, y, model_name):
        """
        Train a machine learning model with hyperparameter optimization
        
        Args:
            X (pd.DataFrame or np.array): Features
            y (pd.Series or np.array): Target
            model_name (str): Name of the model to train
            
        Returns:
            model: Trained model with optimized parameters
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            from sklearn.model_selection import RandomizedSearchCV

            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not supported")

            # 获取参数空间
            param_space = self.param_spaces.get(model_name, {})

            # 使用RandomizedSearchCV进行随机搜索
            if param_space:
                scoring = 'neg_root_mean_squared_error' if self.metric == 'rmse' else 'r2'
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_space,
                    n_iter=self.n_iter,
                    cv=self.cv_folds,
                    scoring=scoring,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                random_search.fit(X, y)
                model = random_search.best_estimator_
                model.best_params_ = random_search.best_params_

            # 添加特征名称处理
            if hasattr(model, 'feature_names_in_'):
                try:
                    model.feature_names_in_ = X.columns.tolist()
                except AttributeError:
                    pass
            
            model.fit(X, y)
            logger.info(f"Successfully trained model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}", exc_info=True)
            raise
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': sqrt(mse),
            'r2': r2
        }
    
    def train_test_split(self, X, y):
        """
        Split data into training and test sets.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
    def train_and_evaluate(self, X, y):
        """
        Train and evaluate all models.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            list: List of dictionaries with model results
        """
        import logging
        logger = logging.getLogger(__name__)
        
        results = []
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = self.train_test_split(X, y)
            
            # Define scoring metric for cross-validation
            if self.metric == 'r2':
                scoring = 'r2'
            else:  # rmse
                scoring = make_scorer(lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)))
            
            # Train and evaluate each model
            for model_name, model in self.models.items():
                try:
                    # Train model
                    trained_model = self.train_model(X_train, y_train, model_name)
                    
                    # Evaluate on test set
                    eval_metrics = self.evaluate_model(trained_model, X_test, y_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        trained_model, X, y,  # 使用训练后的模型
                        cv=self.cv_folds,
                        scoring=scoring
                    )
                    
                    # Calculate mean CV score
                    cv_score = np.mean(cv_scores)
                    if self.metric == 'rmse':
                        cv_score = cv_score  # Convert back from negative RMSE
                    
                    # Evaluate on training set
                    train_metrics = self.evaluate_model(trained_model, X_train, y_train)
                    
                    # Store results
                    # 确保使用正确的键名访问评估指标
                    metric_key = self.metric
                    # 修改结果存储，仅保留cv_score
                    result = {
                        'model_name': model_name,
                        # 移除原始模型对象
                        'model_type': model_name,  # 改为存储模型类型名称
                        'test_score': eval_metrics[metric_key],
                        'cv_score': cv_score,
                        'train_score': train_metrics[metric_key]
                    }
                    
                    results.append(result)
                    logger.info(f"Successfully trained and evaluated model: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Error training model {model_name}: {str(e)}", exc_info=True)
                    continue
            
            # 修改排序逻辑，仅使用cv_score
            if self.metric == 'r2':
                # 对于R2，值越大越好
                results.sort(key=lambda x: x['cv_score'], reverse=True)
            else:
                # 对于RMSE，值越小越好
                results.sort(key=lambda x: x['cv_score'], reverse=False)
            
            # 存储最佳模型（基于综合评分）
            if results:
                # 存储最佳模型对象而非字典，以便可以直接调用predict方法
                best_model_name = results[0]['model_name']
                # 重新训练最佳模型使用全部数据
                self.best_model = self.train_model(X, y, best_model_name)
                logger.info(f"Selected best model: {best_model_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in train_and_evaluate: {str(e)}", exc_info=True)
            raise
    
    def get_best_model(self):
        """
        Get the best model after training.
        
        Returns:
            model: Best trained model
        """
        return self.best_model
