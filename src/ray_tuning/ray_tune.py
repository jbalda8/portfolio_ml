from typing import Dict, Union, Optional
from importlib import import_module

import ray
from ray import tune
from ray.air import session
from ray.tune import ResultGrid


class RayTune:
    """Perform distributed hyperparameter optimization via ray tune

    This class performs distributed and/or sequential based hyperparemeter 
    optimization using Ray Tune: A Python library/collection of hyperparameter 
    optimization techniques that can be used for a wide range of machine 
    learning frameworks. This class has two methods:
        1) objective() - The training procedure for each trial within a       
                         hyperparameter tuning job (cycle)

        2) tuner() - The tuning procedure that runs all tuning trials using the 
                     objective function
    
    Ray tune allows the user to run concurrent trials, utilizing available 
    computing power. Ray creates a "cluster", that is initialized, which sets 
    up the necessary requirements to run tuning jobs. This cluster can take up 
    as much computing power as is avaiable and specified by the user.
    
    Tune provides a function "with_resources" where you can specify how much 
    computing power you would like to use for each individual trial. One 
    caveat, there are some models that you need to manually specify the trial 
    computing power within the model creation. Usually models that are not 
    built in with Ray Tune. You must also specify the use of a GPU within the 
    respective model (if using a GPU). You can do this by passing in those 
    parameters in the search space config, in this case with only 1 option

    More information can be found here: https://docs.ray.io/en/latest/tune/index.html

    Args:
        search_alg: the search algorithm (package) to use for searching  
                    through the hyperparameter space. Most packages use 
                    algorithms such as random search or bayesian optimization

        space: the search space used to sample hyperparameters. Ray Tune 
               provides an easy way to use this search space with it's 
               "with_parameters" method. You can specify different 
               distributions or value sets using tune api. Full examples/
               possibilities can be found here: https://docs.ray.io/en/latest/tune/api/search_space.html#tune-search-space

        data: a dictionary of the most useful requirements/optional information 
              to pass into the objective function. Information can be described 
              as follows:
                  train_data (required): a tuple of training data, e.g. 
                                         (X_train, y_train)

                  validation_data (optional): a tuple of validation data, e.g. 
                                              (X_val, y_val). NOTE: Only needed 
                                              if using validation data to 
                                              compute the score (result of each 
                                              trial), otherwise leave out
                                    
                  model_module (required): string name of the module for which 
                                           the machine learning model comes 
                                           from, e.g. "catboost"

                  model_class_str (required): string name of the model class of 
                                              the machine learning model, e.g. 
                                              CatBoostClassifier or 
                                              CatBoostRegressor

                  fit_params (optional): dictionary of the fit parameters to    
                                         pass into the model at run time (used inside the .fit() method)

                  metric_class_str (required): string name of the Scikit-Learn  
                                               Function excluding the "metrics" 
                                               portion (not the Scoring name), 
                                               e.g. "accuracy_score". Defined 
                                               in https://scikit-learn.org/stable/modules/model_evaluation.html

                  probability (required): boolean value to determine if metric 
                                          requires predict_proba() (set to True), otherwise False

    Returns:
        results: the output of tuner.fit() -> ResultGrid. Contains tuning 
                 results/information
            
    """

    def __init__(self,
                 search_algorithm,
                 search_space: Dict,
                 data: Dict) -> None:
        
        self.search_alg = search_algorithm
        self.space: Dict = search_space
        self.data: Dict = data

    @staticmethod
    def objective(config: Dict, data: Dict) -> None:
        """Definition of objective function used to train each tuning trial

        Args:
            config: a single sampled hyperparameter set to be used for the  
                    current trial. Passed in as **config to imported model

            data: a dictionary of the most useful requirements/optional
                  information to pass into the objective function. Full 
                  information can be found in the class docstring

        Returns:
            None
                
        """

        # Train data - Required
        train_data = data.get('train_data')
        # Validation data, if exists. None otherwise
        try:
            validation_data = data.get('validation_data')
        except KeyError:
            validation_data = None

        # Allow for dynamic model definitions
        model_module = import_module(data.get('model_module'))
        model_class_str = data.get('model_class_str')
        model_class = getattr(model_module, model_class_str)

        # Set the model parameters based on the config
        model = model_class(**config)

        # Train the model
        model.fit(train_data[0],
                  train_data[1],
                  **data.get('fit_params'))

        # Import Scikit-Learn Metric - TODO: Make work for other packages (or 
        # custom metrics)
        metric_module = import_module('sklearn.metrics')
        metric_class_str = data.get('metric_class_str')
        metric_class = getattr(metric_module, metric_class_str)

        # Probability required (e.g. logloss uses predict_proba() so True, F1 uses predict() so False)
        probability = data.get('probability')

        # Score from metric is based on validation if data exists
        if validation_data:
            # predict_proba for probability metrics (e.g. logloss)
            if probability:
                y_pred = model.predict_proba(validation_data[0])
                score = metric_class(validation_data[1], y_pred)
            # Other metrics use predict (class prediction)
            else:
                y_pred = model.predict(validation_data[0])
                score = metric_class(validation_data[1], y_pred)
        # Else score from metric is based on train data
        else:
            # predict_proba for probability metrics (e.g. logloss)
            if probability:
                y_pred = model.predict_proba(train_data[0])
                score = metric_class(train_data[1], y_pred)
            # Other metrics use predict (class prediction)
            else:
                y_pred = model.predict(train_data[0])
                score = metric_class(train_data[1], y_pred)

        # Report score to Ray Tune Session            
        session.report({metric_class_str: score, "done": True})

    def tuner(self,
              init_config: Dict,
              max_concurrent_trials: int,
              num_samples: int,
              cpu_per_trial: int,
              gpu_per_trail: Optional[Union[float, int]] = None) -> ResultGrid:
        """Tuning procedure

        Args:
            init_config: dictionary passed in to ray.init(). Most useful   
                         parameters include:
                             num_cpus (optional): number (int) of cpus to use 
                             in the cluster
                
                             num_gpus (optional): number (int) of gpus to use 
                             in the cluster 

                         Full documentation: https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html

            max_concurrent_trials: the maximum number of parallel trials per 
                                   tuning job

            num_samples: the total number of trials per tuning job

            cpu_per_trial: number of cpus to use for each trial

            gpu_per_trail: proportion of gpu(s) to use for each trial

            NOTE: cpu_per_trial * max_concurrent_trials <= num_cpus
                  gpu_per_trial * max_concurrent_trials <= num_gpus

        Returns:
            results: ray tune ResultGrid type that holds tuning job information 
                     and results
                
        """

        # Initialize ray cluster
        ray.init(**init_config)

        # Define objective function and computing power per trial
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