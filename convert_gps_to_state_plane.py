import pyproj
from pyproj import Proj, transform
import pandas as pd
import numpy as np
import torch
from curvilinear_homography import Curvilinear_Homography

from i24_database_api.db_writer import DBWriter
from i24_configparse import parse_cfg
import matplotlib.pyplot as plt
 

import warnings
warnings.filterwarnings("ignore")



########################### CHANGE ME ##########################################



y_valid = [-150,150]
x_valid = [0,23000]
ms_cutoff =  1000000000000

if False: # Monday, Nov 13
    gps_data_file = "/home/derek/Data/CIRCLES_GPS_NOV_13.csv"
    runID = "6372a3f5577c159a38cbeaf8"
    start_ts = 1668427200
    end_ts   = 1668427200 + (60*60*4)
 
if False: # Tuesday, Nov 14
    start_ts = 1668513600
    end_ts = start_ts + 60*60*4
    gps_data_file = "/home/derek/Data/CIRCLES_GPS_NOV_14.csv"
    runID = "6373d1c840527bf2daa59328"
    
if False:#True: # Wednesday, Nov 15
    start_ts = 1668600000
    end_ts = start_ts + 60*60*4
    gps_data_file = "/home/derek/Data/CIRCLES_GPS_NOV_16.csv"
    runID = "637517698b5b68fc4fd40c77"   
    
if True:#True: # Wednesday, Nov 15
    start_ts = 1668686400
    end_ts = start_ts + 60*60*4
    gps_data_file = "/home/derek/Data/CIRCLES_GPS_NOV_17.csv"
    runID = "6376625b40527bf2daa5932c"


collection_name = "{}_CIRCLES_GPS".format(runID)

feet_per_meter = 3.28084


############################### DONT CHANGE ME ################################



# inProj = Proj(init="epsg:4326")#Proj(proj='utm',zone=16,ellps='WGS84', preserve_units=False)
# outProj = Proj(init='epsg:2274',preserve_units = True)
# x1,y1 = 36.045412, -86.659166
# #x1,y1 = 0,0
# x2,y2 = transform(inProj,outProj,x1,y1)

# wgs84=pyproj.CRS("EPSG:4326")
# tnstate=pyproj.CRS("epsg:2274")
# test_point = 36.0137613,-86.6198052
# out = pyproj.transform(wgs84,tnstate, test_point[0],test_point[1])
# assert np.abs(out[0] - 1785196.31367677) < 2 and  	np.abs(612266.981120024 - out[1]) < 2

def WGS84_to_TN(points):
    """
    Converts GPS coordiantes (WGS64 reference) to tennessee state plane coordinates (EPSG 2274).
    Transform is expected to be accurate within ~2 feet
    
    points array or tensor of size [n_pts,2]
    returns out - array or tensor of size [n_pts,2]
    """
    
    wgs84=pyproj.CRS("EPSG:4326")
    tnstate=pyproj.CRS("epsg:2274")
    out = pyproj.transform(wgs84,tnstate, points[:,0],points[:,1])
    out = np.array(out).transpose(1,0)
    
    if type(points) == torch.Tensor:
        out = torch.from_numpy(out)
        
    return out

# test_point = torch.tensor([36.0137613,-86.6198052]).unsqueeze(0).expand(100,2)
# output = WGS84_to_TN(test_point)








# 1. Initialize homography object
hg = Curvilinear_Homography(save_file = "P01_P40a.cpkl" , downsample = 2)

# 2. Load raw GPS data and assemble into dict with one key per object
# each entry will be array of time, array of x, array of y, array of acc, vehID
vehicles = {}

# TODO - get data from file
dataframe = pd.read_csv(gps_data_file,delimiter = "\t")

ts   = dataframe["systime"].tolist()
ts = [(item/1000 if item > ms_cutoff else item) for item in ts]
lat  = dataframe["latitude"].tolist()
long = dataframe["longitude"].tolist()
vel  = dataframe["velocity"].tolist()
acc  = dataframe["acceleration"].tolist()
vehid   = dataframe["veh_id"].tolist()
acc_setting = dataframe["acc_speed_setting"].tolist()

ts = np.array(ts) #- (6*60*60) # convert from ms to s, then do UTC to CST offset
lat = np.array(lat)
long = np.array(long)
vel = np.array(vel) * feet_per_meter
acc = np.array(acc) * feet_per_meter
vehid = np.array(vehid)
# stack_data
data = np.stack([ts,lat,long,vel,acc,acc_setting,vehid]).transpose(1,0)

# sort by timestamp
data = data[data[:,0].argsort(),:]

# get unique vehids
ids = np.unique(data[:,6])

# assemble into dictionary
vehicles = dict([(id,[]) for id in ids])
for row in data:
    if row[0] < start_ts or row[0] > end_ts:
        continue
    id = row[6]
    vehicles[id].append(row)

    
# lastly, stack each vehicle 
new_vehicles = {}
for key in vehicles.keys():
    if len(vehicles[key]) == 0:
        continue
    data = np.stack(vehicles[key])
    new_data = {
        "ts":data[:,0],
        "lat":data[:,1],
        "long":data[:,2],
        "vel":data[:,3],
        "acc":data[:,4]
        }
    new_vehicles[key] = new_data


vehicles = new_vehicles

# 3. Convert data into roadway coordinates
trunc_vehicles = {}
for vehid in vehicles.keys():
    
    #print("Converting vehicle {}".format(vehid))
    
    data = vehicles[vehid]
    
    # get data as roadway coordinates
    gps_pts  = torch.from_numpy(np.stack([data["lat"],data["long"]])).transpose(1,0)
    deriv_data = torch.from_numpy(np.stack([data["vel"],data["acc"],data["ts"]])).transpose(1,0)
    
    state_pts = WGS84_to_TN(gps_pts)
    state_pts = torch.cat((state_pts,torch.zeros(state_pts.shape[0],1)),dim = 1).unsqueeze(1)
    roadway_pts = hg.space_to_state(state_pts)
    
    veh_counter = 0
    cur_veh_data = [],[] # for xy and tva
    for r_idx in range (len(roadway_pts)):
        row = roadway_pts[r_idx]
        deriv_row = deriv_data[r_idx]
        
        if row[0] > x_valid[0] and row[0] < x_valid[1] and row[1] > y_valid[0] and row[1] < y_valid[1]:
            cur_veh_data[0].append(row)
            cur_veh_data[1].append(deriv_row)
        
        else:
            # break off trajectory chunk
            if len(cur_veh_data[0]) > 30:
                this_road = torch.stack(cur_veh_data[0])
                this_deriv = torch.stack(cur_veh_data[1])
                
                sub_key = "{}_{}".format(vehid,veh_counter)
                trunc_vehicles[sub_key] =   {
                 "x": this_road[:,0],
                 "y": this_road[:,1],
                 "vel":this_deriv[:,0],
                 "acc":this_deriv[:,1],
                 "ts" :this_deriv[:,2],
                 "id":vehid,
                 "run":veh_counter
                 }
                
                veh_counter += 1
            cur_veh_data = [],[]
                

    # 3. Trim data that falls near the workable transform edge, slicing into a separate "trajectory" for each unique run
    # mask = (roadway_pts[:,0] > x_valid[0]).int() * (roadway_pts[:,0] < x_valid[1]).int() * (roadway_pts[:,1] > y_valid[0]).int() * (roadway_pts[:,1] < y_valid[1]).int()
    # keep_idx = mask.nonzero().squeeze()    
    # roadway_pts = roadway_pts[keep_idx,:2]
    # deriv_data = deriv_data[keep_idx,:]
    
    
    # mongo document
    # TODO - add a few more fields of note here

min_ts = []
for obj in trunc_vehicles.values():
    min_ts.append(torch.min(obj["ts"]))
    
print(min(min_ts).item())

# plt.figure()

colors = np.zeros([105,3])
colors[103,0] = 1
colors[59,2] = 1

# for key in trunc_vehicles:
#     data = trunc_vehicles[key]
#     vid = int(data["id"])
#     plt.plot(data["ts"] - 1668607200,data["x"], color = colors[vid])
    
# Need to create as global variable so our callback(on_plot_hover) can access
fig = plt.figure()
plot = fig.add_subplot(111)

# create some curves
for key in trunc_vehicles:
    data = trunc_vehicles[key]
    vid = int(data["id"])
    plot.plot(data["ts"],data["x"], color = colors[vid],gid = vid)

def on_plot_hover(event):
    # Iterating over each data member plotted
    for curve in plot.get_lines():
        # Searching which data member corresponds to current mouse position
        if curve.contains(event)[0]:
            print("over %s" % curve.get_gid())
            
fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)           
plt.show()
    
    
if True:   
    config = parse_cfg("DEFAULT", cfg_name = "WriteWrapperConf.config")
    
    
    
    # 4. Initialize DB writer object
    
    param = {
      "default_host"       : config.host,
      "default_port"       : config.port,
      "default_username"   : config.username,
      "readonly_user"      : config.username,
      "default_password"   : config.password,
      "db_name"            : config.db_name,
      "collection_name"    : collection_name,
      "schema_file"        : config.schema_file,
      "server_id"          : "lambda-cerulean",
      "session_config_id"  : runID,
      "process_name": "CIRCLES_GPS_write"
    }
    
    dbw = DBWriter(param,collection_name = collection_name)
    
    
    
    # 5. Write each object to database
    count = 0
    for key in trunc_vehicles.keys():
        obj = trunc_vehicles[key]
        
        direction = torch.sign(obj["x"][-1]- obj["x"][0]).item()

        x = [item.item()  for item in obj["x"]]
        y = [item.item() * -1  for item in obj["y"]]
        v = [item.item()  for item in obj["vel"]]
        a = [item.item()  for item in obj["acc"]]
        t = [item.item() for item in obj["ts"]]
        
        if len(x) == 0:
            continue
        
        # Nissan Rogue 183″ L x 72″ W x 67″ H
        l = 183/12
        w = 72/12
        h = 67/12
        
    
        # convert to document form
        doc = {}
        doc["configuration_id"]        = runID
        doc["local_fragment_id"]       = int(float(key.split("_")[0]))
        doc["compute_node_id"]         = str(int(float(key.split("_")[0])))
        doc["coarse_vehicle_class"]    = 1
        doc["fine_vehicle_class"]      = -1
        doc["timestamp"]               = t
        doc["raw timestamp"]           = t
        doc["first_timestamp"]         = min(t)
        doc["last_timestamp"]          = max(t)
        doc["road_segment_ids"]        = [-1]
        doc["x_position"]              = x
        doc["y_position"]              = y
        doc["starting_x"]              = x[0]
        doc["ending_x"]                = x[-1]
        doc["camera_snapshots"]        = "None"
        doc["flags"]                   = ["CIRCLES_GPS"]
        doc["length"]                  = l
        doc["width"]                   = w
        doc["height"]                  = h
        doc["direction"]               = direction
        
        
        
        
        # insert
        if len(x) > 10:
            dbw.write_one_trajectory(**doc) 
            #print("Wrote veh {} to database".format(key))
            count += 1
    # 6. Do a celebration dance, you're done
    print("WOHOO! Wrote {} vehicles to database".format(count)) 
    
