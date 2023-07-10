import pandas as pd
from fastf1.events import Event


def prepare_race_data(event: Event) -> pd.DataFrame:
    results = (
        event['Race'].results[['DriverNumber', 'Position', 'Points']].copy()
    )

    race_info = event['Race'].event[['Country', 'Location', 'EventName']].copy()
    results[['Country', 'Location', 'EventName']] = race_info.values

    results['DriverNumber'] = results['DriverNumber'].astype(int)
    return results.reset_index(drop=True)
    
    
    