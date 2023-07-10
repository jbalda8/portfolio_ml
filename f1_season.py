from typing import Dict
import datetime
import pandas as pd
from fastf1.events import get_event_schedule, EventSchedule
from fastf1.core import Session

from event_objects import EventObjects


class F1Season:

    def __init__(self, year: int, end_date: datetime) -> None:
        self.year: int = year
        self.end_date: datetime = end_date
        self.full_season: EventSchedule = get_event_schedule(self.year)
        self.valid_season_df: pd.DataFrame = self.get_season_dataframe() 

    def get_season_dataframe(self) -> pd.DataFrame:
        # Currently testing without Sprint races
        season_df = (
            pd.DataFrame(self.full_season)
            .query('EventFormat in ["conventional"]')
        )

        finished_events = (
            season_df.query("Session4DateUtc < @self.end_date")
        )
        return finished_events
    
    def get_season_sessions(self, round_number: int) -> Dict[str, Session]:
        event_class = EventObjects(self.year, round_number)
        event_dict = {
            session_name: session_object
            for session_name, session_object
            in event_class
        }
        return event_dict
    
    def update_season_dataframe(self) -> None:
        self.valid_season_df['SeasonEvents'] = (
            self.valid_season_df['RoundNumber']
            .apply(lambda round_number: self.get_season_sessions(round_number))
        )