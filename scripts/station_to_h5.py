#!/usr/bin/python3

# script to convert stationXML/SCML inventory to PyVoroTomo station h5
# only _station_ level metadata is needed

import sys

if len(sys.argv) < 2:
    print("use: ./station_to_h5.py <fdsnxml or scml file> <h5 output>"); exit()

import obspy
import pandas as pd
import numpy as np

try:
    inv = obspy.read_inventory(sys.argv[1])
except Exception as e:
    print(f"could not read {sys.argv[1]}: {e}")
    exit()


f_pd = pd.DataFrame(columns=['network', 'station', 'latitude', 'longitude', 'elevation', 'starttime', 'endtime'])
f_pd = f_pd.astype({
    'network':    'string',
    'station':    'string',
    'latitude':   'float64',
    'longitude':  'float64',
    'elevation':  'float64',
    'starttime':  'float64',
    'endtime':    'float64',
})

# unfortunately this assumes the metadata is correct!
rows = []
for net in inv:
    for sta in net:
        end = sta.end_date.timestamp if sta.end_date else 4070908800.0 # 2099. note that it's an obspy UTCDT object so just ".timestamp" is correct
        rows.append([net.code, sta.code, sta.latitude, sta.longitude,
                     np.around(sta.elevation/1000, 1), sta.start_date.timestamp, end])


f_pd = pd.DataFrame(rows, columns=['network','station','latitude','longitude','elevation','starttime','endtime'])
f_pd = f_pd.astype({'network':'string','station':'string','latitude':'float64',
                    'longitude':'float64','elevation':'float64','starttime':'float64','endtime':'float64'})


# check for duplicates
f_pd = f_pd.drop_duplicates(subset=['network','station','endtime','longitude'])

# write out
f_pd.to_hdf('stations.h5',key='stations',mode='w',complevel=5, complib="zlib")
