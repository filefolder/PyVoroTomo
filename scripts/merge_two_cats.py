#!/usr/bin/python3

# combine two separate pyvorotomo events h5s

import sys
if len(sys.argv) < 4:
   print("use: $./merge_two_cats.py <inputA.h5> <inputB.h5> <output.h5>")
   exit()

import pandas as pd

fyleA = str(sys.argv[1])
fyleB = str(sys.argv[2])
output_file = str(sys.argv[3])

def combine_catalogs(fileA, fileB, output_file):
    """
    Combine two HDF5 catalogs containing 'events' and 'arrivals' datasets,
    adjusting event_ids to avoid overlap

    Parameters:
    -----------
    fileA : str
        Path to first HDF5 file
    fileB : str
        Path to second HDF5 file
    output_file : str
        Path where combined HDF5 file will be saved
    """
    events_A = pd.read_hdf(fileA, key='events')
    arrivals_A = pd.read_hdf(fileA, key='arrivals')
    events_B = pd.read_hdf(fileB, key='events')
    arrivals_B = pd.read_hdf(fileB, key='arrivals')

    # Calculate offset (max event_id from first dataset + 1)
    offset = events_A['event_id'].max() + 1

    # Adjust event_ids in second dataset
    events_B['event_id'] = events_B['event_id'] + offset
    arrivals_B['event_id'] = arrivals_B['event_id'] + offset

    # Combine
    combined_events = pd.concat([events_A, events_B], ignore_index=True)
    combined_arrivals = pd.concat([arrivals_A, arrivals_B], ignore_index=True)

    # Save
    combined_events.to_hdf(output_file, key='events', mode='w')
    combined_arrivals.to_hdf(output_file, key='arrivals', mode='a')

combine_catalogs(fyleA,fyleB,output_file)
