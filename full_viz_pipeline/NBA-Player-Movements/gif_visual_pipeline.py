import numpy as np
import io
import os
import pandas as pd
from tqdm import tqdm
import re
from Game import Game

def unload_npy(path):
    """
    unload_npy unpacks a given diffuser output npy file returning the (5, 1024, 66) np array within npy file. 
    """
    data = np.load(path)
    #print(data.files)
    pickle_file = data['archive/data.pkl']
    file_obj = io.BytesIO(pickle_file)
    diffuser_output = np.load(file_obj, allow_pickle=True)
 
    return diffuser_output

def preprocess_npy(diffuser_output, traj_index):
    """
    given the diffuser_output np array (5,1024,66) and a traj_index,
    we reformat that trajectory into a proper array to passed into the json.

    we essentially return the properly formatted trajectories with team id and player id
    but those are arbitrary since we don't care about those.
    """

    # right now we aren't using traj_index
    if traj_index == -1:
        test_reshape = diffuser_output.reshape((1024, 11, 6))
    else:
        test_reshape = diffuser_output[traj_index].reshape((1024, 11, 6))
    

    # remove stats and direction from diffuser output array i.e. keep only the first 3 values of array
    moments = []
    sub_moment = [1,1,1,1, None]
    moment_coord = []

    for i in range(len(test_reshape)):
        for j in range(len(test_reshape[i])):
            if j==0:
                sub_moment_coord = [-1,-1] + list(test_reshape[i][j][:3])
                moment_coord.append(sub_moment_coord)
            elif j==1:
                sub_moment_coord = [1610612747, 977] + list(test_reshape[i][j][:3]) 
                moment_coord.append(sub_moment_coord)
            elif j==2:
                sub_moment_coord = [1610612747, 201579] + list(test_reshape[i][j][:3])
                moment_coord.append(sub_moment_coord)
            elif j==3:
                sub_moment_coord = [1610612747, 101150]+ list(test_reshape[i][j][:3])
                moment_coord.append(sub_moment_coord)
            elif j==4:
                sub_moment_coord = [1610612747, 203903]+ list(test_reshape[i][j][:3])
                moment_coord.append(sub_moment_coord)
            elif j==5:
                sub_moment_coord = [1610612747, 1626204]+ list(test_reshape[i][j][:3])
                moment_coord.append(sub_moment_coord)
            elif j==6:
                sub_moment_coord = [1610612760, 201142]+ list(test_reshape[i][j][:3])
                moment_coord.append(sub_moment_coord)
            elif j==7:
                sub_moment_coord = [1610612760, 201566]+ list(test_reshape[i][j][:3])
                moment_coord.append(sub_moment_coord)
            elif j==8:
                sub_moment_coord = [1610612760, 201586]+ list(test_reshape[i][j][:3])
                moment_coord.append(sub_moment_coord)
            elif j==9:
                sub_moment_coord = [1610612760, 203460]+ list(test_reshape[i][j][:3])
                moment_coord.append(sub_moment_coord)
            elif j==10:
                sub_moment_coord = [1610612760, 203500]+ list(test_reshape[i][j][:3])
                moment_coord.append(sub_moment_coord)
	
            sub_moment_coord = []
        sub_moment.append(moment_coord)
        moment_coord = []

        moments.append(sub_moment)
        sub_moment = [1,1,1,1, None]
        
    #print(np.asarray(moments, dtype=object).shape)

    return np.asarray(moments, dtype=object)

# Read in df that read in base json file:
def process_json(processed_moments, df, eventId):
    """
    with properly formatted traj in processed_moments (note: only for 1 traj in diffuser output npy file),
    we append this traj to the BASE df that read in an arbitrarily formatted json file with eventId = eventId

    we return a json string of the properly appended and formatted game with new traj (diffuser output traj)
    """
    appended_event = df.loc[0].copy(deep=True)
    # create a new event (formatted) and add that event to df, save new df as json
    appended_event["events"] = appended_event["events"].copy()
    appended_event["events"]["eventId"] = eventId
    appended_event["events"]["moments"] = processed_moments
#    print(appended_event)
    df = pd.concat([df, pd.DataFrame([appended_event])], ignore_index=True)
#    print(df.tail()["events"])
    json_str = df.to_json()

    return json_str

def visualize(npy_file):

    json_file = "../data/0021500549.json"
    df = pd.read_json(json_file)
    df = df.drop(index=df.index.difference([0]))

    # Split the file_path by "_dir-"
    parts = npy_file.split("_dir-")

    traj_index = 0

    # Check if the split resulted in two parts
    if len(parts) == 2:
        last_part = parts[1]
        digit = last_part.split("-")[0]
        traj_index = digit
    else:
        print("Digit not found in the file path")

    diffuser_output = unload_npy(npy_file)
    if diffuser_output.ndim == 2:
        processed_moments = preprocess_npy(diffuser_output, -1) # c
        eventId = f"999{traj_index}"
        json_str = process_json(processed_moments, df, eventId)
        df = pd.read_json(json_str)
    elif diffuser_output.ndim == 3:
        for traj_index in range(5):
            processed_moments = preprocess_npy(diffuser_output, traj_index) # c
            eventId = f"999{traj_index}"
            json_str = process_json(processed_moments, df, eventId)
            df = pd.read_json(json_str)


    name, extension = os.path.splitext(npy_file)
    if extension:
        name = name + '.json'

    game = Game(json_df=df, path_to_json=name, event_index=9990)
    game.read_json()

    game.start()


    


    
