from f1_season import F1Season
from prepare_data.lap_data import prepare_lap_data
from prepare_data.weather_data import Weather
from prepare_data.control_message_data import prepare_control_message_data
from prepare_data.driver_data import prepare_driver_data
from prepare_data.race_data import prepare_race_data
import pandas as pd

from fastf1.core import Session


for i in [2023]:

    end_date = pd.to_datetime('2023-07-07')
    f1_season = F1Season(i, end_date)
    f1_season.update_season_dataframe()

    for j, event in enumerate(f1_season.valid_season_df['SeasonEvents']):
        results = prepare_race_data(event)
        results['LocalOrder'] = j + 1
        for session in event.items():
            if session[0] != 'Race':
                try:
                    prepared_lap_data = prepare_lap_data(session[1])

                    added_weather_data = Weather(prepared_lap_data, session[1])
                    full_dataset = pd.DataFrame()
                    for row in added_weather_data:
                        full_dataset = pd.concat([full_dataset, row])

                    racer_flags = prepare_control_message_data(session[1])
                    updated_full_dataset = pd.merge(full_dataset, racer_flags, on='DriverNumber', how='left') # Move nans to none (str)

                    driver_data = prepare_driver_data(session[1])
                    # You can merge this for each session in event
                    updated_full_dataset = pd.merge(updated_full_dataset, driver_data, on='DriverNumber', how='left')
                    updated_full_dataset = pd.merge(updated_full_dataset, results, on='DriverNumber', how='inner')

                    updated_full_dataset['SessionType'] = session[0]
                    updated_full_dataset['SeasonYear'] = i 
                    
                    season_df = pd.concat([season_df, updated_full_dataset], ignore_index=True)
                except:
                    print(f"Session: {session[1]}")