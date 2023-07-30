from typing import Dict
import datetime
import pandas as pd
from fastf1.events import get_event_schedule, EventSchedule
from fastf1.core import Session

from src.model_data.season_objects.session_objects import SessionObjects


class F1Season:
    """Obtain fastf1 season dataframe with session objects in each event

    TODO: Decription

    Args:
        year: the year of the season

        end_date: the last day a session could take place on, determined by the 
                  training data cutoff. Only relevant for a season that is 
                  ongoing

    Returns:
        valid_season_df: holds the event info for valid events in a season with 
                         session names and objects within dataframe
            
    """

    def __init__(self, year: int, end_date: datetime) -> None:
        self.year: int = year
        self.end_date: datetime = end_date
        self.full_season: EventSchedule = get_event_schedule(self.year)
        self.valid_season_df: pd.DataFrame = self.get_season_dataframe() 

    def get_season_dataframe(self) -> pd.DataFrame:
        # TODO: Doctring

        # Currently testing without Sprint races
        season_df = (
            pd.DataFrame(self.full_season)
            .query('EventFormat in ["conventional"]')
        )

        # Only obtain finished events, for training data
        finished_events = (
            season_df.query("Session4DateUtc < @self.end_date")
        )

        return finished_events
    
    def get_season_sessions(self, round_number: int) -> Dict[str, Session]:
        # TODO: Doctring

        # Get all sessions in an event
        session_class = SessionObjects(self.year, round_number)
        
        # Returned event dict that holds session names and objects
        event_dict = {
            session_name: session_object
            for session_name, session_object
            in session_class
        }

        return event_dict
    
    def update_season_dataframe(self) -> None:
        # TODO: Doctring

        # Get all sessions for each event in valid season races
        self.valid_season_df['SeasonEvents'] = (
            self.valid_season_df['RoundNumber']
            .apply(lambda round_number: self.get_season_sessions(round_number))
        )
