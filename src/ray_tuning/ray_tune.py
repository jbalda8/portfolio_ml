import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor, Pool

import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session

class RayTune:

    def __init__(self, search_algorithm, search_space, data) -> None:
        self.search_algorithm = search_algorithm
        self.search_space = search_space
        self.data = data

    def objective(self, config, data):

        # Set the CatBoostRegressor parameters based on the config
        model = CatBoostRegressor(
            **config
        )

        try:
            X_val = data.get('X_val')
            y_val = data.get('y_val')
        except KeyError:
            eval_set = None
        else:
            try:
                assert X_val is not None
                assert y_val is not None
                eval_set = (X_val, y_val)
            except:
                eval_set = None

        # Train the model
        model.fit(data.get('X_train'),
                    data.get('y_train'),
                    eval_set=eval_set,
                    **data.get('fit_params'),
                    cat_features=data.get('cat_features'))

        # Evaluate the model
        score = model.get_evals_result()

        metric = data.get('metric')
        metric_lower = metric.lower()

        if eval_set:
            globals()[metric_lower] = score['validation'][metric][-1]
        else:
            globals()[metric_lower] = score['learn'][metric][-1]
            
        session.report({metric_lower: globals()[metric_lower], "done": True})

    def tuner(self):

        trainable_with_cpu_gpu = (
            tune.with_resources(self.objective, {"cpu" : 4, "gpu": 0.2})
        )

        # Create Tuner object
        tuner = tune.Tuner(
            tune.with_parameters(trainable_with_cpu_gpu, data=data),
            tune_config=tune.TuneConfig(
                search_alg=hyperopt_search,
                max_concurrent_trials=5, 
                num_samples=5
            ),
            param_space=space,
        )

        # Fit Tuner
        results = tuner.fit()