import pandas as pd
from fastf1.core import Session


def lap_number_pr(rows):
    row_list = list(rows)
    last_true_index = (
        len(row_list) - row_list[::-1].index(True)
    )
    return last_true_index


def prepare_lap_data(data: Session) -> pd.DataFrame:

    lap_data = data.laps

    date_columns = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']

    for date_column in date_columns:
        lap_data[f'{date_column}Seconds'] = (
            lap_data[date_column].dt.total_seconds()
        )

    lap_data = lap_data.query('IsAccurate == True').sort_values(by=['DriverNumber', 'LapNumber'])

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

    aggregated_lap_data.columns = (
        [f'{col[0]}_{col[1]}' 
         if col[1] != '' else col[0] 
         for col in aggregated_lap_data.columns]
    )
    return aggregated_lap_data