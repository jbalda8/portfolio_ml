from typing import Dict
from catboost import CatBoostRegressor

import ray
from ray import tune
from ray.air import session


class RayTune:
    """Perform distributed hyperparameter optimization via ray tune

    Args:
        search_alg:

        space:

        data:

    Returns:
        results:
            
    """

    def __init__(self,
                 search_algorithm,
                 search_space: Dict,
                 data: Dict) -> None:
        
        self.search_alg = search_algorithm
        self.space: Dict = search_space
        self.data: Dict = data

    @staticmethod
    def objective(config, data) -> None:
        """Definition of objective function used to train each tuning trial

        Args:
            config:

            data:

        Returns:
            None
                
        """

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

    def tuner(self,
              init_config,
              cpu_per_trial,
              gpu_per_trail,
              max_concurrent_trials,
              num_samples):
        """Tuning procedure

        Args:
            init_config:

            cpu_per_trial:

            gpu_per_trail:

            max_concurrent_trials:

            num_samples:

        Returns:
            results:
                
        """

        ray.init(**init_config)

        trainable_with_cpu_gpu = (
            tune.with_resources(self.objective,
                                {"cpu" : cpu_per_trial, 
                                 "gpu": gpu_per_trail})
        )

        # Create Tuner object
        tuner = tune.Tuner(
            tune.with_parameters(trainable_with_cpu_gpu, data=self.data),
            tune_config=tune.TuneConfig(
                search_alg=self.search_alg,
                max_concurrent_trials=max_concurrent_trials, 
                num_samples=num_samples
            ),
            param_space=self.space,
        )

        # Fit Tuner
        results = tuner.fit()
        ray.shutdown()

        return results