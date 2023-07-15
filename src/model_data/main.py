from typing import List, Tuple
import pandas as pd

from src.model_data.f1_season import F1Season
from src.model_data.prepare_data.lap_data import prepare_lap_data
from src.model_data.prepare_data.weather_data import Weather
from src.model_data.prepare_data.control_message_data import prepare_control_message_data
from src.model_data.prepare_data.driver_data import prepare_driver_data
from src.model_data.prepare_data.race_data import prepare_race_data


class RunAllMethods:

    def __init__(self, seasons: List[int], end_date: str) -> None:
        self.seasons = seasons
        self.end_date = pd.to_datetime(end_date)

    def get_next_season(self) -> Tuple[int, List]:
        curr_season = self.seasons.pop(0)

        f1_season = F1Season(curr_season, self.end_date)
        f1_season.update_season_dataframe()

        events = f1_season.valid_season_df['SeasonEvents']
        combined_dict = [(key, value) for values in events 
                         for key, value in values.items()]
        
        return (curr_season, combined_dict)

    def __iter__(self):
        return self
    
    def __next__(self) -> pd.DataFrame:
        if len(self.seasons) == 0:
            raise StopIteration
        
        curr_season, combined_dict = self.get_next_season()
        
        feature_df = pd.DataFrame()
        result_df = pd.DataFrame()

        for session_name, session_object in combined_dict:

            session_object.load()
            # Catch when session cannot be loaded from fastf1
            if not hasattr(session_object, '_laps'):
                continue
            else:
                pass

            if session_object == 'PredictingRace':
                return session_object
            elif session_name == 'Race':
                # Prepare race data (results and information)
                results = prepare_race_data(session_object)
                results['SeasonYear'] = curr_season
                results['EventName'] = session_object.event.EventName
                results['RoundNumber'] = session_object.event.RoundNumber

                # Merge race data
                result_df = (
                    pd.concat([results, result_df])
                )
            else:
                # Prepare lap data
                prepared_lap_data = prepare_lap_data(session_object)

                # Prepare weather data and attach to lap data
                added_weather_data = (
                        Weather(prepared_lap_data, session_object)
                )
                full_dataset = pd.DataFrame()
                for row in added_weather_data:
                    full_dataset = pd.concat([full_dataset, row])

                # Prepare control message data
                racer_flags = (
                        prepare_control_message_data(session_object)
                )
                # Merge control message data with full dataset
                updated_full_dataset = (
                    pd.merge(full_dataset, 
                                racer_flags, 
                                on='DriverNumber', 
                                how='left')
                )

                # Prepare driver data
                driver_data = prepare_driver_data(session_object)
                # Merge driver data
                updated_full_dataset = (
                    pd.merge(updated_full_dataset, 
                                driver_data, 
                                on='DriverNumber', 
                                how='left')
                )

                # Add session information
                updated_full_dataset['SessionType'] = session_name
                updated_full_dataset['SeasonYear'] = curr_season
                updated_full_dataset['EventName'] = (
                    session_object.event.EventName
                )

                feature_df = pd.concat([feature_df, updated_full_dataset])

        merged_df = (
            pd.merge(feature_df,
                     result_df,
                     on=['DriverNumber',
                         'EventName',
                         'SeasonYear'],
                         how='inner')
        )

        return merged_df.reset_index(drop=True)
