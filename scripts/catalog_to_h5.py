#!/usr/bin/python3

#convert a QML or SCML catalog to a pyvorotomo model file

import sys

if len(sys.argv) < 2: print("use: ./cat2pvt.py <qml/sc3ml file or 'FDSN'> <pvt h5 output>"); exit()

catfile = str(sys.argv[1])
h5file  = str(sys.argv[2])

import h5py, obspy, os
import pandas as pd
import numpy as np

### set values here
fdsn_server = "EARTHSCOPE" # or the URL, or http://localhost:80

minpicks = 10
mindist = 0.01 #degrees
maxdist = 2.2 #degrees

minmag = 0.9
mindepth = -1.99 #km
maxdepth= 15
minweight = 0.10 #aka "time weight" here
maxHerr = 8
maxVerr = 8
maxazigap = 300
min_spick_ratio = .2 #ratio of S:P picks
runQC = True #run a secondary QC routine (see code)


#if looking at all origins there may be duplicate picks. prefer the newest ones
def get_latest_picks(picks):
    newpicks = picks.copy()

    #check that creation time is in all picks 1st
    for i,p in enumerate(newpicks):
        if not p.creation_info: #happens!
            p.creation_info = obspy.core.event.base.CreationInfo()
            p.creation_info.creation_time = obspy.UTCDateTime(1980,1,1)

        if not p.creation_info.creation_time:
            p.creation_info.creation_time = obspy.UTCDateTime(1980,1,1)

    for i,p in enumerate(newpicks):
        doubles = [(p.creation_info.creation_time,p.resource_id.id)]
        for j,p_ in enumerate(newpicks[i+1:]):
            if p.waveform_id.network_code == p_.waveform_id.network_code and \
            p.waveform_id.station_code == p_.waveform_id.station_code and \
            p.phase_hint == p_.phase_hint:
                try:
                    doubles.append((p_.creation_info.creation_time,p_.resource_id.id))
                except:
                    print("issue with ", p)
                    continue

        if len(doubles) > 1:
            doubles.sort() #sorts by time..
            to_remove = doubles[:1] #want the newest..
            newpicks = [ele for ele in newpicks if (ele.creation_info.creation_time,ele.resource_id.id) not in to_remove]
    return newpicks

def get_relevant_pick(arrival,picks):
    p = [ele for ele in picks if ele.resource_id == arrival.pick_id]
    return p[0]

def get_relevant_arrival(pick,eq):
    for o in eq.origins:
        for arr in o.arrivals:
           if arr.pick_id == pick.resource_id:
               return arr
    return None

def get_spick_ratio(eq):
    o = eq.origins[0]
    s = [a for a in o.arrivals if a.phase=='S']
    return float( len(s)/len(o.arrivals) )


def is_uncertainty_bad(eq,max_horiz_uncert=15,max_depth_uncert=15,min_rms=0.0,maxdepth=80):
    #if RMS is below some value, let it through regardless
    if eq.origins[0].quality.standard_error < min_rms: return False

    # deeper events are hard to constrain. may want to soften constraints
    if eq.origins[0].depth/1000 > maxdepth:
        max_horiz_uncert *= 2
        max_depth_uncert *= 2

    #see if any component of an earthquake is greater than these limits
    try:
        eq_horiz_error = max( eq.origins[0].latitude_errors.uncertainty, eq.origins[0].longitude_errors.uncertainty)
    except:
        eq_horiz_error = 0
    try:
        eq_vert_error = eq.origins[0].depth_errors.uncertainty/1000
    except:
        eq_vert_error = 0
    if eq_horiz_error <= max_horiz_uncert and eq_vert_error <= max_depth_uncert:
        return False # EVENT IS GOOD
    else:
        return True # EVENT IS BAD




#check if output file exists, if not copy from a blank/skeleton file. note that mix/matching python versions seems to cause issues elsewhere be advised!!
if not os.path.exists(h5file):
    pyvoro_arrivals_h5 = pd.DataFrame(columns = ['network', 'station', 'event_id', 'phase', 'time', 'residual', 'snr'])
    pyvoro_events_h5 = pd.DataFrame(columns = ['event_id', 'latitude', 'longitude', 'depth', 'time', 'residual','source_id']) #added source_id to keep track of them outside of this
else:
    pyvoro_arrivals_h5 = pd.read_hdf(h5file,key='arrivals') #arrivals / [network, station, event_id, phase, time, residual, snr]
    pyvoro_events_h5 = pd.read_hdf(h5file,key='events')




#can also just load one in via FDSN...
if catfile.upper() == "FDSN":
    print("loading catalog from FDSN...")

    cat = obspy.core.event.Catalog()
    from obspy.clients.fdsn import Client
    event_client = Client(fdsn_server)

    t0 = obspy.UTCDateTime(2024,9,1); t1 = obspy.UTCDateTime(2026,1,10); num_split=100


    t_array = np.linspace(t0.timestamp,t1.timestamp,num_split)
    for i in range(len(t_array)-1):
        _t0 = obspy.UTCDateTime(t_array[i]); _t1 = obspy.UTCDateTime(t_array[i+1])
        print("searching %s - %s" % (obspy.UTCDateTime(t_array[i]), obspy.UTCDateTime(t_array[i+1])), end='')
        try:
            cat0 = event_client.get_events(starttime=t_array[i],endtime=t_array[i+1],
                                mindepth=mindepth,
                                minlat=32,maxlat=37,
                                minlon=-121,maxlon=-115,
                                maxdepth=maxdepth,
                                minmagnitude=minmag,
                                includearrivals=True,
                                includeallmagnitudes=False) #socal

            if runQC:
                try:
                    # normal qc
                    cat0.events = [eq for eq in cat0.events
                                   if len(eq.origins[0].arrivals) >= minpicks
                                   and not is_uncertainty_bad(eq,maxHerr,maxVerr,0)
                                   and eq.origins[0].quality.azimuthal_gap < maxazigap
                                   and get_spick_ratio(eq) >= min_spick_ratio]
                except Exception as e:
                    print(" issue with event filtering.. ", e)
                    cat0.events = [eq for eq in cat0.events if eq.origins[0].quality.standard_error is not None]
                    cat0.events = [eq for eq in cat0.events
                                   if len(eq.origins[0].arrivals) >= minpicks
                                   and not is_uncertainty_bad(eq,maxHerr,maxVerr,0)
                                   and eq.origins[0].quality.azimuthal_gap < maxazigap
                                   and get_spick_ratio(eq) >= min_spick_ratio]
            cat += cat0
            print(" ...got %d good events" % (len(cat0)))
        except Exception as e:
            print(e)
            pass
else:
    cat = obspy.read_events(catfile)
    print("%s loaded" % catfile)


print("total number of events: %d" % len(cat))



from tqdm import tqdm
events_to_add = []
phases_to_add = []
for voro_i,eq in enumerate(tqdm(cat)):

    try: o = eq.origins[0]
    except:
        print("no origin for ", eq)
        continue
    if o.depth is None: continue # skip any events without a depth

    try: rms = o.quality.standard_error
    except: rms = 1
    
    try: mag = eq.magnitudes[0].mag
    except: mag = 0.999
    
    try: azigap = o.quality.azimuthal_gap
    except: azigap = 0
    
    try:
        v_uncert = o.depth_errors.uncertainty/1000 #KM
        h_uncert = (o.latitude_errors.uncertainty + o.longitude_errors.uncertainty)/2 #KM
    except:
        v_uncert = 0; h_uncert = 0
    
    if v_uncert > 100:
        print("wrong vertical uncert given for ", eq)
        continue
    
    #if o.depth < 0: o.depth = -0.5  #reset air-quakes unless we have a good reason not to. 
    event_row = { 'event_id': voro_i, 'latitude':o.latitude, 'longitude': o.longitude, 'depth':np.around(o.depth/1000,3), 'time':o.time.timestamp, 'residual':rms,
                  'mag': mag, 'azigap':azigap, 'v_uncert':v_uncert, 'h_uncert': h_uncert, 'source_id':str(eq.resource_id)} #added h/v uncert, azigap
    events_to_add.append(event_row)

    picks = get_latest_picks(eq.picks)
    if len(picks) == 0: continue
    if o.creation_info.agency_id == 'WEL(GNS_Primary)': continue
    if o.quality.azimuthal_gap > maxazigap: continue

    for p in picks:
        a = get_relevant_arrival(p,eq)
        if a is None:
            print("no associated arrival found???")
            continue

        if not a.time_weight: a.time_weight = 1.01*minweight

        # recall that distance is DEGREES
        if a.distance is None or a.time_weight < minweight: continue

        ## over-ride weight and residuals. let pyvorotomo figure it out
        if no_weights:
            a.time_weight = 1
            a.time_residual = 1

        phasehint = p.phase_hint[0].upper()
        phase_row = {'network':p.waveform_id.network_code, 'station':p.waveform_id.station_code, 'event_id': voro_i, \
                     'phase':phasehint, 'time':p.time.timestamp, 'residual':a.time_residual, 'snr':a.time_weight, 'distance':a.distance}
        phases_to_add.append(phase_row)



#now dump all at once instead of piecewise
pyvoro_arrivals_h5 = pd.DataFrame(phases_to_add)
pyvoro_events_h5 = pd.DataFrame(events_to_add)

# set NaNs to something
pyvoro_events_h5 = pyvoro_events_h5.fillna(0)
pyvoro_arrivals_h5 = pyvoro_arrivals_h5.fillna(2)


pyvoro_events_h5.to_hdf(h5file,key='events',mode='w',complevel=5, complib="zlib")
pyvoro_arrivals_h5.to_hdf(h5file,key='arrivals',mode='a',complevel=5, complib="zlib")

print("%s written" % h5file)
