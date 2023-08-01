# Formula 1 Race Predictor

Catboost model that predicts the points a driver will get in a given Formula 1 race. Please also see machine_learning_full_lifecycle.ipynb for the full analysis/predictions.

## Table of Contents

- [Overview](#overview)
- [Process](#process)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Overview

The "Formula 1 Race Predictor" is an advanced machine learning model designed to predict the points a driver will earn in a Formula 1 race based on session-level data. This model aims to address the challenge of accurately forecasting a driver's performance in a dynamic and highly competitive racing environment.

The primary purpose of the "Formula 1 Race Predictor" is to hypothetically enhance race strategizing and decision-making for teams and drivers in the Formula 1 championship. By leveraging historical session-level data, the model provides valuable insights into a driver's potential race performance.

Predicting a driver's points in a Formula 1 race is a complex and multifaceted problem due to the multitude of variables influencing race outcomes. The "Formula 1 Race Predictor" model addresses this challenge by analyzing various performance indicators, such as lap times, sector times, speed statistics, and race conditions. By fusing these diverse data points, the model generates accurate predictions that can aid teams in maximizing their drivers' competitive edge.

The "Formula 1 Race Predictor" stands out from traditional predictive models due to its unique integration of session-level data, which captures fine-grained details of a driver's performance during different phases of a race weekend. The model leverages this granular information to identify subtle patterns and trends that impact a driver's point-scoring potential.

## Process

1. Pull data using procedure in *src/model_data* module
2. Preprocess and prepare data for analysis
3. Hyperparameter tuning using *src/ray_tuning* module
4. Training and evaluation

## Features

Data is typically shown on an aggregated level since each driver has 1 row per session. For example, a driver has many laps in a session, so aggregated information about those laps is computed (min, max, mean, etc.). Other data includes driver information, weather information, season information, etc.

### Driver Information:
- `Driver`: Abbreviated name of the Formula 1 driver.
- `DriverNumber`: The unique driver number associated with the driver.
- `CountryCode`: The driver's nationality
- `TeamId`: Unique identifier for the team of the driver.

### Lap Time Statistics:
- `LapTimeSeconds`: Lap time achieved by the driver during the session. (Min, Max, Mean, Std, Count - number of laps in session)
- `IsPersonalBest_pr_lap`: The lap number for the driver's personal best lap during the given session

### Sector Time Statistics:
- `Sector1TimeSeconds`: Time taken to complete Sector 1 of the race track. (Min, Max, Mean, Std)
- `Sector2TimeSeconds`: Time taken to complete Sector 2 of the race track. (Min, Max, Mean, Std)
- `Sector3TimeSeconds`: Time taken to complete Sector 3 of the race track. (Min, Max, Mean, Std)

### Speed Statistics:
- `SpeedI1`: Speed achieved in sector 1 during the session. (Min, Max, Mean, Std)
- `SpeedI2`: Speed achieved in sector 2 during the session. (Min, Max, Mean, Std)
- `SpeedFL`: Speed achieved at finish line. (Min, Max, Mean, Std)
- `SpeedST`: Speed achieved on longest straight (Min, Max, Mean, Std)

### Weather Information:
- `AirTemp`: Air temperature during the time the driver was on the track. (Min, Max, Mean, Std)
- `Humidity`: Humidity level during the time the driver was on the track. (Min, Max, Mean, Std)
- `Pressure`: Atmospheric pressure during the time the driver was on the track. (Min, Max, Mean, Std)
- `TrackTemp`: Track temperature during the time the driver was on the track. (Min, Max, Mean, Std)
- `WindDirection`: Wind direction during the time the driver was on the track. (Min, Max, Mean, Std)
- `WindSpeed`: Wind speed during the time the driver was on the track. (Min, Max, Mean, Std)

### Additional Information:
- `Category`: Any control message information (events) for a driver during a session. Examples include CarEvent and Flag.
- `SessionType`: Type of the race session (e.g., Qualifying, Practice, etc.).
- `SeasonYear`: Year of the racing season.
- `EventName`: Name of the race event.
- `Country`: Country where the race event takes place.
- `Location`: Specific location or circuit of the race event (city).
- `RoundNumber`: Round number of the race event in a season.

## Installation

To set up the required environment for your project using Conda, follow these steps:

1. **Clone Repository:** Clone portfolio_ml repo into local directory

   ```bash
   git clone https://github.com/jbalda8/portfolio_ml.git
   ```

2. **Install Conda:** If you don't have Conda installed, download and install Anaconda or Miniconda by following the instructions for your operating system:

   - [Anaconda](https://www.anaconda.com/products/individual)
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

3. **Create a New Environment:** Open a terminal or command prompt and run the following command to create a new environment named "formula1_predictor" (you can choose a different name if you prefer inside environment.yml):

   ```bash
   conda env create -f environment.yml
   ```

## Usage

To use the "Formula 1 Race Predictor" machine learning model, follow these steps:

1. **Set Up Environment:** Ensure you have set up the required environment for the project as described in the [Installation](#installation) section. Activate the environment using the appropriate command:

    ```bash
    conda activate formula1_predictor
    ```

2. **Prepare Data:** Ensure you have the data for your Formula 1 sessions in a structured format (e.g., CSV, Excel, or DataFrame). The dataset should contain the features specified in the [Features](#features) section, such as driver statistics, lap times, weather conditions, and other relevant information. There are example datasets in the *data* folder. You can also reference the **Pull Data** section in the *machine_learning_full_lifecycle.ipynb* notebook for an example on how to obtain data using the *src/model_data* module.

3. **Data Preprocessing:** Before feeding the data into the model, perform necessary data preprocessing steps. This may involve handling missing values, feature scaling, encoding categorical variables, and splitting the data into training and testing sets.

4. **Hyperparameter Tuning:** Before training the final model and evaluating the performance, it is helpful to tune the hyperparameters for the model of choice. There is a procedure set up in the *src/ray_tuning* module that uses Ray Tune: A distributed hyperparameter tuning package. A helpful visual provided by Ray Tune to describe to process:

![Alt text](https://docs.ray.io/en/latest/_images/tune_flow.png)

You can find a full example in the Hyperparameter Optimization section in the *machine_learning_full_lifecycle.ipynb* notebook.

5. **Model Training:** Train the "Formula 1 Race Predictor" model using the preprocessed data. Import the necessary libraries and load the dataset. Split the data into features (X) and the target variable (y). Instantiate the model and fit it to the training data.

   ```python
   # Example Python code for model training
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from catboost import CatBoostRegressor

   # Load the preprocessed data
   data = pd.read_csv('your_preprocessed_data.csv')

   # Split the data into features (X) and target variable (y)
   X = data.drop(columns=['Points'])
   y = data['Points']

   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   params = {
      # Define parameter set here - From tuning
   }

   # Instantiate and train the model using optimized parameter set
   model = CatBoostRegressor(**params)
   model.fit(X_train, y_train)

## Evaluation

The performance of the "Formula 1 Race Predictor" model is evaluated using several evaluation metrics to assess its predictive accuracy and generalization capabilities. The following evaluation metrics are commonly used:

- **Mean Absolute Error (MAE):** This metric measures the average absolute difference between the predicted points and the actual points. A lower MAE indicates better predictive accuracy.

- **Mean Squared Error (MSE):** MSE calculates the average squared difference between the predicted points and the true points. It penalizes larger errors more than MAE, making it sensitive to outliers.

- **R-squared (R2):** Also known as the coefficient of determination, R-squared measures the proportion of the variance in the target variable that is predictable from the independent features. A higher R2 value (close to 1) indicates a better fit of the model to the data.

During the model's development, these metrics are computed on the test set to determine how well the model generalizes to unseen data. The evaluation results play a crucial role in fine-tuning hyperparameters and selecting the best model configuration.

## Results

The "Formula 1 Race Predictor" model has shown promising results in predicting the points a driver will get in a race based on session-level data. Here are some key findings:

- The Mean Squared Error (MSE) is X, suggesting the model effectively captures variations in points across different races.

Feature Importance (Top 20 Features): 

|    | Feature                 |   Importance |
|---:|:------------------------|-------------:|
|  0 | TeamId                  |    39.4998   |
|  1 | SeasonYear              |    12.939    |
|  2 | Driver                  |     7.21664  |
|  3 | SpeedI1_min             |     2.92208  |
|  4 | Sector1TimeSeconds_max  |     2.37767  |
|  5 | LapTimeSeconds_max      |     2.36861  |
|  6 | SpeedI2_mean            |     2.25792  |
|  7 | SpeedFL_std             |     1.66686  |
|  8 | Pressure_min            |     1.32463  |
|  9 | Sector3TimeSeconds_max  |     1.23987  |
| 10 | Sector3TimeSeconds_std  |     1.14842  |
| 11 | Sector1TimeSeconds_min  |     1.11676  |
| 12 | RoundNumber             |     0.962125 |
| 13 | Sector2TimeSeconds_mean |     0.940766 |
| 14 | SpeedST_mean            |     0.915074 |
| 15 | SpeedST_std             |     0.905343 |
| 16 | Sector2TimeSeconds_std  |     0.876212 |
| 17 | WindSpeed_std           |     0.858126 |
| 18 | Sector1TimeSeconds_std  |     0.767986 |
| 19 | LapTimeSeconds_min      |     0.758734 |

Example race results with predictions:

|    | Driver   | EventName        |   SeasonYear |   Position |   Points |   PredictedPoints |
|---:|:---------|:-----------------|-------------:|-----------:|---------:|------------------:|
|  0 | VER      | Miami Grand Prix |         2023 |          1 |       26 |         17.5999   |
|  1 | PER      | Miami Grand Prix |         2023 |          2 |       18 |         17.2954   |
|  2 | HAM      | Miami Grand Prix |         2023 |          6 |        8 |         14.9895   |
|  3 | RUS      | Miami Grand Prix |         2023 |          4 |       12 |         11.576    |
|  4 | LEC      | Miami Grand Prix |         2023 |          7 |        6 |         10.7393   |
|  5 | SAI      | Miami Grand Prix |         2023 |          5 |       10 |          9.38954  |
|  6 | NOR      | Miami Grand Prix |         2023 |         17 |        0 |          3.95676  |
|  7 | OCO      | Miami Grand Prix |         2023 |          9 |        2 |          3.04603  |
|  8 | GAS      | Miami Grand Prix |         2023 |          8 |        4 |          2.66433  |
|  9 | PIA      | Miami Grand Prix |         2023 |         19 |        0 |          1.68342  |
| 10 | DEV      | Miami Grand Prix |         2023 |         18 |        0 |          1.41777  |
| 11 | ALO      | Miami Grand Prix |         2023 |          3 |       15 |          0.852914 |
| 12 | BOT      | Miami Grand Prix |         2023 |         13 |        0 |          0.777002 |
| 13 | STR      | Miami Grand Prix |         2023 |         12 |        0 |          0.728528 |
| 14 | TSU      | Miami Grand Prix |         2023 |         11 |        0 |          0.573643 |
| 15 | MAG      | Miami Grand Prix |         2023 |         10 |        1 |          0.492269 |
| 16 | ALB      | Miami Grand Prix |         2023 |         14 |        0 |          0.368515 |
| 17 | SAR      | Miami Grand Prix |         2023 |         20 |        0 |          0.311086 |
| 18 | HUL      | Miami Grand Prix |         2023 |         15 |        0 |          0.275934 |
| 19 | ZHO      | Miami Grand Prix |         2023 |         16 |        0 |          0.274426 |

Full analysis can be seen in *machine_learning_full_lifecycle.ipynb*

## Future Work

Listed below are some potential future enhancements and tasks for the "Formula 1 Race Predictor" machine learning model:

- **Data Class of Ray Tune** I have created a data class in *src/ray_tuning/data.py* that needs to be implemented in ray_tune.py in order to use my custom overfitting metric (optionally) instead of an sklearn metric.

- **Feature Engineering:** Continuously investigate new features that could enhance the model's understanding of driver performance during sessions.

- **Feature Selection:** Use feature selection ideology to hone down on the number of features used in training and scoring.

- **F1 Betting:** Get an idea of how the model would perform when placing bets on race results.

## License

The "Formula 1 Race Predictor" model is released under the [MIT License](LICENSE). You are free to use, modify, and distribute the model as per the terms specified in the license.

Please refer to the [LICENSE](LICENSE) file for more details on the permissions and limitations of the MIT License.
