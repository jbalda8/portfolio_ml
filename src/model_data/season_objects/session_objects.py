from typing import List, Union, Tuple
import re
import pandas as pd
from fastf1.events import Event, get_event
from fastf1 import set_log_level
from fastf1.core import Session


set_log_level("ERROR") # Set fastf1 logging to errors only

class SessionObjects:
    """Obtain fastf1 Session objects for a event in a season

    This class is of type iterator and obtains Session objects for each event 
    in a F1 season. The valid events in an F1 season are determined in the 
    f1_season.py file. Each session in an event is tagged and used to obtain 
    data (from files in prepare_data folder)

    Args:
        year: the year of the season in which the event takes place

        gp: stands for Grand Prix and is either the name of the Grand Prix in 
            full or the race rank in a season (i.e. what order the race is in 
            the season)

    Returns:
        session_name: the name of the session, e.g. Practice 1
        
        session_object: of fastf1 type Session, which holds all data for the 
                        given session specified
            
    """

    def __init__(self, year: int, gp: Union[int, str]) -> None:
        self.event: Event = get_event(year, gp)
        self.session_names: List[str] = self.get_session_names()
        self.current_index: int = 0

    def get_session_names(self) -> List[str]:
        """Get a list of session names in a given event

        Event object returns a callable data storage of session information 
        within the event. This method pulls the name of the sessions to aid in 
        the process of pulling each session object

        Args:
            None

        Returns:
            session_names_list: list of session names, e.g. ['Practice 1', 
                                'Practice 2', 'Practice 3', 'Qualifying']

        """

        # Pattern to detect sessions
        pattern = r"^Session\d+$"

        # Obtain session names that meet given contraints
        session_names_list = [
            session[1]
            for session in self.event.items()
            if re.match(pattern, session[0]) is not None
            and session[1] != ''
        ]

        return session_names_list
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[str, Session]:
        # TODO: Docstring

        # Stop iteration criteria
        if self.current_index >= len(self.session_names):
            raise StopIteration    

        # Get current session name
        session_name = self.session_names[self.current_index]
        splits = session_name.split(' ')

        # Logic to obtain session type and number if relevant
        # Sprint logic is included, although may not be used
        if splits[0] == 'Sprint' and len(splits) > 1:
            session_type = splits[0].lower() + '_' + splits[1].lower()
            session_round = None
        elif len(splits) > 1:
            session_type, session_round = splits
        else:
            session_type = splits[0]
            session_round = None

        # Obtain the fastf1 function to get session data. Either 
        # get_qualifying, get_practice, get_sprint, or get_sprint_shootout
        function_name = f"get_{session_type.lower()}"
        if hasattr(self.event, function_name):
            session_func = getattr(self.event, function_name)

        # Will have to change get_season_dataframe() method in f1_season.py if  
        # Sprint races are desired. Currently only pulls conventional races
        if session_round and session_type not in ['Sprint', 'Sprint Shootout']:
            session_object = session_func(session_round)
        else:
            session_object = session_func()

        # Go to next index
        self.current_index += 1

        return session_name, session_object
    