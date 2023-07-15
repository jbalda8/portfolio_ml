from typing import Dict
import pandas as pd
from fastf1.events import Session


def prepare_race_data(data: Session) -> pd.DataFrame:
    """Prepare race data for a given event

    This function obtains both the event descriptions, i.e. Country where the 
    race took place, Grand Prix, etc., and potential response variables 
    "Points" or "Position"

    Args:
        data: passed in as a fastf1 Session type. This datatype includes 
              all possible information on the given session. This function uses 
              it's "results" method to obtain result data and "event" method to 
              obtain event information

    Returns:
        results: race result data for each driver and event descriptions. 
                 Returns a pandas dataframe
        
    """

    # Include response variables and driver identification
    results = (
        data.results[['DriverNumber', 'Position', 'Points']].copy()
    )

    # Pull in race descriptions
    race_info = data.event[['Country', 'Location', 'EventName']].copy()
    results[['Country', 'Location', 'EventName']] = race_info.values

    # Keep identifier name and datatype consistent with rest of data
    results['DriverNumber'] = results['DriverNumber'].astype(int)

    return results.reset_index(drop=True)
    