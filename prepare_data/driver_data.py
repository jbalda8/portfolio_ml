import pandas as pd
from fastf1.core import Session


def prepare_driver_data(data: Session) -> pd.DataFrame:
    """Prepare driver data for a given session

    Driver data includes demographic information for each driver, along with team information. This function is used for each individual session. Although it is unlikely that this information will change through a F1 season, it has happened before (in the case of a F1 team changing drivers midway through the season). Therefore, to avoid any issues/missing information, this function is kept on the session grain.

    Args:
        data: passed in as a fastf1 Session datatype. This datatype includes all possible information on the given session. This function uses it's get_driver() method to obtain control message data

    Returns:
        full_driver_data: data for each driver within a session. Returns a pandas dataframe:
            DriverNumber: the unique driver number for a given driver in the session
            TeamId: the team identification in which the driver belongs to
            CountryCode: the country of origin for a given driver
    
    """

    # Currently only uses TeamId and CountryCode for the model
    # DriverNumber is for indentification
    full_driver_data = (
        pd.DataFrame(columns=['DriverNumber', 'TeamId', 'CountryCode'])
    )

    # Uses `get_driver` method to obtain driver info for each driver 
    for driver_number in data.drivers:
        driver_data = data.get_driver(driver_number)
        select_driver_df = (
            pd.DataFrame(driver_data[[
                'DriverNumber', 
                'TeamId', 
                'CountryCode']]).T
        )
        # Concat each additional driver row to full dataset
        full_driver_data = pd.concat([select_driver_df, full_driver_data])

    full_driver_data['DriverNumber'] = (
        full_driver_data['DriverNumber'].astype(int)
    )
    return full_driver_data.reset_index(drop=True)