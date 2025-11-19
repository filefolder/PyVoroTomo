#!/usr/bin/python3

#remove events from <input A> which are no longer in <input b>

import sys
if len(sys.argv) < 3:
   print("use: $./remove_from.py <inputA.h5> <inputB.h5>")
   exit()

import pandas as pd

remove_arrivals_also = True


fyleA = str(sys.argv[1])
fyleB = str(sys.argv[2])
eventsA = pd.read_hdf(fyleA,key='events')
eventsB = pd.read_hdf(fyleB,key='events')
arrivalsA = pd.read_hdf(fyleA, key='arrivals')
arrivalsB = pd.read_hdf(fyleB, key='arrivals')

# Filter eventsA to keep only those events still present in B after however much QC
eventsA_filtered = eventsA[eventsA['event_id'].isin(eventsB['event_id'])]

# Filter arrivalsA to keep only arrivals that exist in arrivalsB
# Note that this will make man events fall below the minimum arrivals filter
if remove_arrivals_also:
    arrivalsA_filtered = arrivalsA.merge(
        arrivalsB[['event_id', 'station', 'phase']],
        on=['event_id', 'station', 'phase'],
        how='inner'
    )
else:
    arrivalsA_filtered = arrivalsB

num_removed = len(eventsA)-len(eventsA_filtered)
num_arrivals_removed = len(arrivalsA)-len(arrivalsA_filtered)
if num_removed > 0 or num_arrivals_removed > 0:
        with pd.HDFStore(fyleA+".new", mode='w') as store:
            store.put('events', eventsA_filtered)
            store.put('arrivals', arrivalsA_filtered)

print("removed %d events" % num_removed)
print("removed %d arrivals" % num_arrivals_removed)
