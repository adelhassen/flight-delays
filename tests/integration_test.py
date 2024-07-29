import requests

flight_info = {'AIRLINE': 'Endeavor Air Inc.',
 'ORIGIN': 'AUS',
 'DEST': 'RDU',
 'CRS_ELAPSED_TIME': 185.0,
 'DAY_OF_MONTH_SIN': 0.20129852008866006,
 'DAY_OF_MONTH_COS': 0.9795299412524945,
 'DAY_OF_WEEK_SIN': -0.7818314824680299,
 'DAY_OF_WEEK_COS': 0.6234898018587334,
 'CRS_DEP_TIME_MINUTES_SIN': 0.4733196671848435,
 'CRS_DEP_TIME_MINUTES_COS': -0.8808907382053855,
 'CRS_ARR_TIME_MINUTES_SIN': -0.5446390350150271,
 'CRS_ARR_TIME_MINUTES_COS': -0.838670567945424}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=flight_info)

# Script successfully connected to Docker and returned a result
assert isinstance(response, requests.models.Response)