import pandas as pd
from fastf1.core import Session


def prepare_control_message_data(data: Session) -> pd.DataFrame:
    """Prepare control message data for a given F1 session

    Control messages for a F1 session incude a number of descriptive events that occur within a session. These messages are sent to all teams to keep them informed on session status or cautions. Control messages can be tagged to a driver, in the event that they had some car related event or even a potential crash. These messages can also be unrelated to a specific driver, althogh this function focuses on the control messages that are related to a specific driver

    Args:
        data: passed in as a fastf1 Session type. This datatype includes all possible information on the given session. This function uses it's "race_control_messages" method to obtain control message data

    Returns:
        racer_flags: any flags related to a driver. Returns a pandas dataframe:
            DriverNumber: the unique driver number for a given driver in the session
            Category: the category of the control message, which can be one of the followinge ['Other', 'Flag', 'Drs', 'CarEvent']
            
    """

    # Pull control message data in which a racer is involved 
    control_message_data = data.race_control_messages
    racer_flags = (
        control_message_data
        .query('RacingNumber.notnull()')[['RacingNumber', 'Category']]
        .reset_index(drop=True)
    )

    # Keep identifier name and datatype consistent with rest of data
    racer_flags = (
        racer_flags.rename(columns={'RacingNumber': 'DriverNumber'})
    )
    racer_flags['DriverNumber'] = racer_flags['DriverNumber'].astype(int) 
    
    return racer_flags
