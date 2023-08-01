from dataclasses import dataclass
from typing import Optional


@dataclass
class Data:
    """ Definition of data class to pass into RayTune class

    train_data (required): a tuple of training data, e.g. 
                           (X_train, y_train)

    model_module (required): string name of the module for which 
                             the machine learning model comes 
                             from, e.g. "catboost"
                
    model_class_str (required): string name of the model class of 
                                the machine learning model, e.g. 
                                CatBoostClassifier or 
                                CatBoostRegressor
                    
    metric_class_str (required): NOTE - Only use if use_overfitting_metric == 
                                 False. String name of the Scikit-Learn function excluding the "metrics" 
                                 portion (not the Scoring name), e.g. 
                                 "accuracy_score". Defined in https://scikit-learn.org/stable/modules/model_evaluation.html
                            
    probability (required): boolean value to determine if metric 
                            requires predict_proba() (set to 
                            True), otherwise False

    validation_data (optional): a tuple of validation data, e.g. 
                                (X_val, y_val). NOTE: Only needed 
                                if using validation data to 
                                compute the score (result of each 
                                trial), otherwise leave out

    fit_params (optional): dictionary of the fit parameters to    
                           pass into the model at run time (used 
                           inside the .fit() method)

    metric_params (optional): dictionary of additional parameters 
                              to pass in to Sklearn metric chosen

    metric_name (optional): optional name for metric in objective 
                            output. If not used, name will be 
                            metric_class_str

    use_overfitting_metric (optional): boolean indication on whether the    
                                       predefined overfitting_metric() in RayTune class should be used to score each tuning run

    """
    
    train_data: tuple
    model_module: str
    model_class_str: str
    metric_class_str: str
    probability: bool
    validation_data: Optional[tuple] = None
    fit_params: Optional[dict] = None
    metric_params: Optional[dict] = None
    metric_name: Optional[str] = None
    use_overfitting_metric: Optional[bool] = False

    def __post_init__(self):
        if (
            (self.use_overfitting_metric is True) &
            ((self.metric_class_str is not None) or
            (self.metric_params is not None))
        ):
            raise ValueError("""If using overfitting metric, do not pass in metric_class_str or metric_params.""")
        
        if not isinstance(self.train_data, tuple):
            raise TypeError(f"Expected 'train_data' to be tuple, but got {type(self.train_data).__name__}")
        if not isinstance(self.model_module, str):
            raise TypeError(f"Expected 'model_module' to be str, but got {type(self.model_module).__name__}")
        if not isinstance(self.model_class_str, str):
            raise TypeError(f"Expected 'model_class_str' to be str, but got {type(self.model_class_str).__name__}")
        if not isinstance(self.metric_class_str, str):
            raise TypeError(f"Expected 'metric_class_str' to be str, but got {type(self.metric_class_str).__name__}")
        if not isinstance(self.probability, bool):
            raise TypeError(f"Expected 'probability' to be bool, but got {type(self.probability).__name__}")
        if self.validation_data is not None and not isinstance(self.validation_data, tuple):
            raise TypeError(f"Expected 'validation_data' to be tuple, but got {type(self.validation_data).__name__}")
        if self.fit_params is not None and not isinstance(self.fit_params, dict):
            raise TypeError(f"Expected 'fit_params' to be dict, but got {type(self.fit_params).__name__}")
        if self.metric_params is not None and not isinstance(self.metric_params, dict):
            raise TypeError(f"Expected 'metric_params' to be dict, but got {type(self.metric_params).__name__}")
        if self.metric_name is not None and not isinstance(self.metric_name, str):
            raise TypeError(f"Expected 'metric_name' to be str, but got {type(self.metric_name).__name__}")
        if self.use_overfitting_metric is not None and not isinstance(self.use_overfitting_metric, bool):
            raise TypeError(f"Expected 'use_overfitting_metric' to be bool, but got {type(self.use_overfitting_metric).__name__}")