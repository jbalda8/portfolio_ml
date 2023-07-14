from src.model_data.driver import RunAllMethods


seasons = [2022, 2023]
end_date = '2023-07-12'

driver_class = RunAllMethods(seasons, end_date)
season_df = driver_class.__next__()