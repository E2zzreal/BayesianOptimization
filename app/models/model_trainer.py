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
        best_models_params = {} # Store best params found for each model on full data

        try:
            # Define scoring metric based on self.metric for RandomizedSearchCV
            if self.metric == 'r2':
                scoring = 'r2'
            elif self.metric == 'rmse':
                # Use scikit-learn's built-in neg_root_mean_squared_error
                scoring = 'neg_root_mean_squared_error'
            else:
                # Fallback or raise error for unsupported metrics
                scoring = 'r2' # Defaulting to r2, consider raising error
                logger.warning(f"Unsupported metric '{self.metric}', defaulting to 'r2' for scoring.")

            # Train and evaluate each model using RandomizedSearchCV on the full dataset
            for model_name, base_model in self.models.items():
                try:
                    logger.info(f"Starting RandomizedSearchCV for {model_name} on the full dataset...")
                    from sklearn.model_selection import RandomizedSearchCV

                    # Get param space for the current model
                    param_space = self.param_spaces.get(model_name, {})

                    if not param_space:
                         logger.warning(f"No hyperparameter space defined for {model_name}. Skipping RandomizedSearchCV, using default parameters.")
                         # If no param space, train with default params and estimate CV score differently?
                         # For now, we'll proceed but CV score might not be comparable if search wasn't done.
                         # Option: Perform simple cross_val_score with default model
                         cv_scores = cross_val_score(base_model, X, y, cv=self.cv_folds, scoring=scoring, n_jobs=-1)
                         cv_score = np.mean(cv_scores)
                         best_params = base_model.get_params() # Use default params
                         # Need to handle negative scores if using neg_rmse
                         if scoring == 'neg_root_mean_squared_error':
                             cv_score = -cv_score
                    else:
                        # Perform Randomized Search CV on the *entire* dataset (X, y)
                        random_search = RandomizedSearchCV(
                            estimator=base_model,
                            param_distributions=param_space,
                            n_iter=self.n_iter,
                            cv=self.cv_folds, # Inner CV loop for hyperparameter tuning
                            scoring=scoring,
                            random_state=self.random_state,
                            n_jobs=-1,
                            refit=False # We don't need the refitted model from here, just the score and params
                        )
                        random_search.fit(X, y)

                        # Use the best score found during the search as the CV score
                        cv_score = random_search.best_score_
                        best_params = random_search.best_params_

                        # Handle negative scores if using neg_rmse or similar
                        if scoring == 'neg_root_mean_squared_error':
                            cv_score = -cv_score # Convert back to positive RMSE

                        logger.info(f"RandomizedSearchCV completed for {model_name}. Best CV Score ({self.metric}): {cv_score:.4f}")

                    # Store results - only model name and its CV score
                    result = {
                        'model_name': model_name,
                        'cv_score': cv_score,
                        # Store best params to train the final model later
                        'best_params': best_params
                    }
                    results.append(result)
                    best_models_params[model_name] = best_params # Keep track of best params

                except Exception as e:
                    logger.error(f"Error during RandomizedSearchCV for model {model_name}: {str(e)}", exc_info=True)
                    continue

            # Sort results based on the CV score (best_score_ from RandomizedSearchCV)
            if not results:
                 logger.error("No models were successfully evaluated.")
                 return []

            if self.metric == 'r2':
                # 对于R2，值越大越好
                results.sort(key=lambda x: x['cv_score'], reverse=True)
            else:
                # 对于RMSE，值越小越好
                results.sort(key=lambda x: x['cv_score'], reverse=False)
            
            # Select the best model based on the CV score
            if results:
                best_result = results[0]
                best_model_name = best_result['model_name']
                best_model_params = best_models_params.get(best_model_name, {})

                logger.info(f"Selected best model based on CV score: {best_model_name} with score {best_result['cv_score']:.4f}")

                # Train the final best model on the *entire* dataset using the best hyperparameters found
                final_model = self.available_models[best_model_name] # Get a fresh instance
                try:
                    if best_model_params:
                         final_model.set_params(**best_model_params)
                    logger.info(f"Training final best model ({best_model_name}) on the full dataset...")
                    final_model.fit(X, y)
                    self.best_model = final_model
                    logger.info(f"Final best model ({best_model_name}) trained successfully.")
                except Exception as e:
                    logger.error(f"Error training final best model {best_model_name}: {str(e)}", exc_info=True)
                    self.best_model = None # Ensure best_model is None if final training fails

            # Return results containing only model name and cv_score
            # Remove best_params from the returned list as it's internal use for final training
            final_results = [{'model_name': r['model_name'], 'cv_score': r['cv_score']} for r in results]
            return final_results
            
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
