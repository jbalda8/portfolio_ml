import pandas as pd
import numpy as np
from fastf1.core import Session
from typing import Tuple


def lap_number_pr(pr_bool: np.array) -> int:
    """Obtain the lap number for a driver's session personal best

    The IsPersonalBest field in session.laps is a boolean field that marks whether a lap was a personal best at the time of that lap finishing. This means that for each driver in a session, you could have multiple True indications. This function returns the best lap number overall, i.e. the last lap where a lap is marked as a Personal Best lap

    Args:
        pr_bool: bool array of IsPersonalBest for driver's laps

    Returns:
        last_true_index: lap number for driver's Personal Best
        
    """
    
    # Convert array to list
    row_list = list(pr_bool)

    try:
        # Find last index of personal best lap
        last_true_index = (
            len(row_list) - row_list[::-1].index(True)
        )
        return last_true_index
    except:
        return -1


def prepare_lap_data(data: Session) -> pd.DataFrame:
    """Prepare lap data for a given session

    Lap data includes an exhaustive list of aggregated lap related fields, e.g. lap time overall, lap time at each sector, speeds, and personal best lap number. The aggregations include common metrics like min, max, average, and standard deviation. Not everything may be useful, but it would be easy to obtain feature importance and analyze which of these metrics are most useful in predicting how a racer is doing/will do 

    Args:
        data: passed in as a fastf1 Session type. This datatype includes all possible information on the given session. This function uses it's "laps" method to obtain lap data

    Returns:
        aggregated_lap_data: data for each driver within a session with their laps information. Returns a pandas dataframe

    """

    # Pull lap data
    lap_data = data.laps

    # Convert time columns to seconds (easier to work with) 
    time_columns = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
    for time_column in time_columns:
        lap_data[f'{time_column}Seconds'] = (
            lap_data[time_column].dt.total_seconds()
        )

    # Only include accurate (valid) laps
    lap_data = lap_data.sort_values(by=['DriverNumber', 'LapNumber'])

    # Specify which metrics to pull for each column
    lap_agg_dict = {
        'Time': ['min', 'max'],
        'LapTimeSeconds': ['min', 'max', 'mean', 'std', 'count'],
        'Sector1TimeSeconds': ['min', 'max', 'mean', 'std'],
        'Sector2TimeSeconds': ['min', 'max', 'mean', 'std'],
        'Sector3TimeSeconds': ['min', 'max', 'mean', 'std'],
        'SpeedI1': ['min', 'max', 'mean', 'std'],
        'SpeedI2': ['min', 'max', 'mean', 'std'],
        'SpeedFL': ['min', 'max', 'mean', 'std'],
        'SpeedST': ['min', 'max', 'mean', 'std'],
        'IsPersonalBest': [('pr_lap', lap_number_pr)]
    }

    # Obtain aggregates for each driver and column
    aggregated_lap_data = (
        lap_data[
            ['Driver',
             'DriverNumber',
             'Time', 
             'LapTimeSeconds',
             'Sector1TimeSeconds',
             'Sector2TimeSeconds',
             'Sector3TimeSeconds',
             'SpeedI1',
             'SpeedI2',
             'SpeedFL',
             'SpeedST',
             'IsPersonalBest']]
             .groupby(['Driver', 'DriverNumber'])
             .agg(lap_agg_dict)
             .reset_index()
    )

    # Restructure columns
    aggregated_lap_data.columns = (
        [f'{col[0]}_{col[1]}' 
         if col[1] != '' else col[0] 
         for col in aggregated_lap_data.columns]
    )

    return aggregated_lap_data
