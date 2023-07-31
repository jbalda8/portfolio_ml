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

2. **Prepare Data:** Ensure you have the data for your Formula 1 sessions in a structured format (e.g., CSV, Excel, or DataFrame). The dataset should contain the features specified in the [Features](#features) section, such as driver statistics, lap times, weather conditions, and other relevant information. There are example datasets in the *data* folder. You can also reference the **Pull Data** section in the *machine_learning_full_lifecycle.ipynb" notebook for an example on how to pull data.

3. **Data Preprocessing:** Before feeding the data into the model, perform necessary data preprocessing steps. This may involve handling missing values, feature scaling, encoding categorical variables, and splitting the data into training and testing sets. You can reference the *machine_learning_full_lifecycle.ipynb* for an example using catboost.

4. **Model Training:** Train the "Formula 1 Race Predictor" model using the preprocessed data. Import the necessary libraries and load the dataset. Split the data into features (X) and the target variable (y). Instantiate the model and fit it to the training data.

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

   # Instantiate and train the model
   model = CatBoostRegressor()
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

|    | Feature                |   Importance |
|---:|:-----------------------|-------------:|
|  0 | TeamId                 |     32.3625  |
|  1 | Driver                 |      9.61222 |
|  2 | SeasonYear             |      4.3194  |
|  3 | SpeedFL_std            |      1.71069 |
|  4 | SpeedST_std            |      1.70791 |
|  5 | SpeedI1_min            |      1.70208 |
|  6 | Sector1TimeSeconds_min |      1.68767 |
|  7 | SpeedST_min            |      1.6198  |
|  8 | SpeedFL_min            |      1.57691 |
|  9 | SpeedI1_mean           |      1.47165 |
| 10 | SpeedI1_std            |      1.38573 |
| 11 | TrackTemp_mean         |      1.34834 |
| 12 | WindSpeed_mean         |      1.31693 |
| 13 | Sector2TimeSeconds_std |      1.20522 |
| 14 | SpeedFL_mean           |      1.20281 |
| 15 | WindDirection_mean     |      1.18462 |
| 16 | SpeedI2_mean           |      1.13946 |
| 17 | TrackTemp_std          |      1.13023 |
| 18 | Sector1TimeSeconds_max |      1.10384 |
| 19 | SpeedI2_std            |      1.0973  |

Example race results with predictions:

|    | Driver   | EventName        |   SeasonYear |   Position |   Points |   PredictedPoints |
|---:|:---------|:-----------------|-------------:|-----------:|---------:|------------------:|
|  0 | VER      | Miami Grand Prix |         2023 |          1 |       26 |         18.0725   |
|  1 | HAM      | Miami Grand Prix |         2023 |          6 |        8 |         15.8886   |
|  2 | RUS      | Miami Grand Prix |         2023 |          4 |       12 |         13.8039   |
|  3 | PER      | Miami Grand Prix |         2023 |          2 |       18 |         13.5481   |
|  4 | LEC      | Miami Grand Prix |         2023 |          7 |        6 |          9.89182  |
|  5 | SAI      | Miami Grand Prix |         2023 |          5 |       10 |          9.71395  |
|  6 | NOR      | Miami Grand Prix |         2023 |         17 |        0 |          6.58196  |
|  7 | ALO      | Miami Grand Prix |         2023 |          3 |       15 |          2.59517  |
|  8 | GAS      | Miami Grand Prix |         2023 |          8 |        4 |          1.86957  |
|  9 | OCO      | Miami Grand Prix |         2023 |          9 |        2 |          1.69518  |
| 10 | PIA      | Miami Grand Prix |         2023 |         19 |        0 |          1.19671  |
| 11 | BOT      | Miami Grand Prix |         2023 |         13 |        0 |          0.83791  |
| 12 | STR      | Miami Grand Prix |         2023 |         12 |        0 |          0.667938 |
| 13 | DEV      | Miami Grand Prix |         2023 |         18 |        0 |          0.636996 |
| 14 | TSU      | Miami Grand Prix |         2023 |         11 |        0 |          0.590246 |
| 15 | HUL      | Miami Grand Prix |         2023 |         15 |        0 |          0.528032 |
| 16 | ALB      | Miami Grand Prix |         2023 |         14 |        0 |          0.475799 |
| 17 | MAG      | Miami Grand Prix |         2023 |         10 |        1 |          0.314835 |
| 18 | ZHO      | Miami Grand Prix |         2023 |         16 |        0 |          0.295501 |
| 19 | SAR      | Miami Grand P

Full analysis can be seen in *machine_learning_full_lifecycle.ipynb*

## Future Work

Listed below are some potential future enhancements and tasks for the "Formula 1 Race Predictor" machine learning model:

- **Feature Engineering:** Continuously investigate new features that could enhance the model's understanding of driver performance during sessions.

- **Feature Selection:** Use feature selection ideology to hone down on the number of features used in training and scoring.

- **F1 Betting:** Get an idea of how the model would perform when placing bets on race results.

## License

The "Formula 1 Race Predictor" model is released under the [MIT License](LICENSE). You are free to use, modify, and distribute the model as per the terms specified in the license.

Please refer to the [LICENSE](LICENSE) file for more details on the permissions and limitations of the MIT License.
