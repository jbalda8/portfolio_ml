from typing import List
import pandas as pd
from fastf1.core import Session


class Weather:
    """Prepare weather data for a given session

    This class obtains aggregated weather data for each driver during the time in which they were on the track. This class is of type iterator that can be used to obtain all weather data for a given session. Weather data includes known desciptives like: temperature, windspeed, pressure, etc. An exhaustive aggregation is computed by using metrics like: max, min, average, and standard deviation. 
    
    It is originally unknown which weather fields are most important in aiding in the determination of a racer's chance of winning, but further analysis can be conducted via feature importance to determine this matter

    Args:
        lap_data: the already prepared lap data from lap_data.py. This way weather data can just be appended to the end of each driver's session lap aggregations

        data: passed in as a fastf1 Session type. This datatype includes all possible information on the given session. This function uses it's "weather_data" method to obtain weather related data

    Returns:
        results: race result data for each driver and event descriptions. Returns a pandas dataframe
        
    """

    def __init__(self, lap_data_prepared: pd.DataFrame, data: Session) -> None:
        self.lap_data: pd.DataFrame = lap_data_prepared
        self.weather_data: pd.DataFrame = data.weather_data
        self.lap_data_indices: List = list(self.lap_data.index)

    def weather_for_racer(self, row: object) -> pd.DataFrame:
        """Get a single row of weather data for a given driver in a given session

        Args:
            row: lap data as a single row already computed for driver

        Returns:
            output: single row of inclusion of weather data to a driver's lap data. Returns a pandas dataframe

        """

        driver_number = row['DriverNumber']

        # Beginning of a driver's session
        start_time = (
            self.lap_data
            .query('DriverNumber == @driver_number')['Time_min']
            .values[0]
        )

        # Ending of a driver's session
        end_time = (
            self.lap_data
            .query('DriverNumber == @driver_number')['Time_max']
            .values[0]
        )

        # Aggregation metrics per feature
        weather_agg_dict = {
        'AirTemp': ['min', 'max', 'mean', 'std'],
        'Humidity': ['min', 'max', 'mean', 'std'],
        'Pressure': ['min', 'max', 'mean', 'std'],
        'TrackTemp': ['min', 'max', 'mean', 'std'],
        'WindDirection': ['min', 'max', 'mean', 'std'],
        'WindSpeed': ['min', 'max', 'mean', 'std'],
        }

        # Obtain aggregated weather data
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

        # Output and labels to join into lap data
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
        # If no more driver lap data, end iterator
        if len(self.lap_data_indices) == 0:
            raise StopIteration
        
        # Take top row index
        index = self.lap_data_indices.pop(0)

        # Row as object type and dataframe type
        row_object = self.lap_data.iloc[index]
        row_dataframe = pd.DataFrame(self.lap_data.iloc[index]).T
        
        # Keep dataframe index as original index
        row_dataframe.index = [row_object.name]

        # Obtain row data for this row using row object
        weather_row = self.weather_for_racer(row_object)
        full_row = pd.concat([row_dataframe, weather_row], axis=1)

        # Keep identifier name and datatype consistent with rest of data
        full_row['DriverNumber'] = full_row['DriverNumber'].astype(int)
        
        return full_row
        