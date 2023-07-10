from typing import List, Union, Tuple
import re
import pandas as pd
from fastf1.events import Event, get_event
from fastf1 import set_log_level
from fastf1.core import Session

set_log_level("ERROR")

class EventObjects:

    def __init__(self, year: int, gp: Union[int, str]) -> None:
        self.event: Event = get_event(year, gp)
        self.session_names: List[str] = self.get_session_names()
        self.current_index: int = 0

    def get_session_names(self) -> List[str]:
        pattern = r"^Session\d+$"

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
        if self.current_index >= len(self.session_names):
            raise StopIteration    

        session_name = self.session_names[self.current_index]
        splits = session_name.split(' ')

        if splits[0] == 'Sprint' and len(splits) > 1:
            session_type = splits[0].lower() + '_' + splits[1].lower()
            session_round = None
        elif len(splits) > 1:
            session_type, session_round = splits
        else:
            session_type = splits[0]
            session_round = None

        function_name = f"get_{session_type.lower()}"

        if hasattr(self.event, function_name):
            session_func = getattr(self.event, function_name)

        if session_round and session_type not in ['Sprint', 'Sprint Shootout']:
            session_object = session_func(session_round)
        else:
            session_object = session_func()

        now_utc = pd.Timestamp.utcnow().to_datetime64()
        if session_object.date < now_utc:
           session_object.load()
        else:
            session_object = 'Predicting Race'

        self.current_index += 1

        return session_name, session_object