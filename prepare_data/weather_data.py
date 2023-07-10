from typing import List
import pandas as pd
from fastf1.core import Session


class Weather:

    def __init__(self, lap_data_prepared: pd.DataFrame, data: Session) -> None:
        self.lap_data: pd.DataFrame = lap_data_prepared
        self.weather_data: pd.DataFrame = data.weather_data
        self.lap_data_indices: List = list(self.lap_data.index)

    def weather_for_racer(self, row: object) -> pd.DataFrame:
        driver_number = row['DriverNumber']
        start_time = (
            self.lap_data
            .query('DriverNumber == @driver_number')['Time_min']
            .values[0]
        )
        end_time = (
            self.lap_data
            .query('DriverNumber == @driver_number')['Time_max']
            .values[0]
        )

        weather_agg_dict = {
        'AirTemp': ['min', 'max', 'mean', 'std'],
        'Humidity': ['min', 'max', 'mean', 'std'],
        'Pressure': ['min', 'max', 'mean', 'std'],
        'TrackTemp': ['min', 'max', 'mean', 'std'],
        'WindDirection': ['min', 'max', 'mean', 'std'],
        'WindSpeed': ['min', 'max', 'mean', 'std'],
        }

        aggregated_weather_data = (
            self.weather_data
            .query('@start_time <= Time <= @end_time')[
                ['AirTemp', 
                 'Humidity',
                 'Pressure',
                 'TrackTemp',
                 'WindDirection',
                 'WindSpeed']].apply(weather_agg_dict)
        )

        # Reshape the DataFrame into a single row
        single_row = (
            aggregated_weather_data
            .melt(ignore_index=False)
            .reset_index()
        )

        index = row.name
        output = pd.DataFrame({
            f"{variable}_{index}": value
            for index, variable, value
            in zip(
                single_row['index'],
                single_row['variable'],
                single_row['value'])}, index=[index])

        return output
    
    def __iter__(self):
        return self
    
    def __next__(self) -> pd.DataFrame:
        if len(self.lap_data_indices) == 0:
            raise StopIteration
        
        index = self.lap_data_indices.pop(0)

        row_object = self.lap_data.iloc[index]
        row_dataframe = pd.DataFrame(self.lap_data.iloc[index]).T
        row_dataframe.index = [row_object.name]

        weather_row = self.weather_for_racer(row_object)
        full_row = pd.concat([row_dataframe, weather_row], axis=1)

        full_row['DriverNumber'] = full_row['DriverNumber'].astype(int)
        
        return full_row
        

