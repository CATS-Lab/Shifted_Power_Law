import pandas as pd
import torch
import numpy as np
import torch.utils.data as data
import os
from tqdm import tqdm


from .read_highD import read_track_csv


# TRACK FILE
BBOX = "bbox"
FRAMES = "frames"
FRAME = "frame"
TRACK_ID = "id"
X = "x"
Y = "y"
WIDTH = "width"
HEIGHT = "height"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
FRONT_SIGHT_DISTANCE = "frontSightDistance"
BACK_SIGHT_DISTANCE = "backSightDistance"
DHW = "dhw"
THW = "thw"
TTC = "ttc"
PRECEDING_X_VELOCITY = "precedingXVelocity"
PRECEDING_ID = "precedingId"
FOLLOWING_ID = "followingId"
LEFT_PRECEDING_ID = "leftPrecedingId"
LEFT_ALONGSIDE_ID = "leftAlongsideId"
LEFT_FOLLOWING_ID = "leftFollowingId"
RIGHT_PRECEDING_ID = "rightPrecedingId"
RIGHT_ALONGSIDE_ID = "rightAlongsideId"
RIGHT_FOLLOWING_ID = "rightFollowingId"
LANE_ID = "laneId"




# Load data
class highD(data.Dataset):
    def __init__(self, data_root,
                 loc_id = 1,
                 output_steps = 25,
                 train_ratio = 0.7,
                 train = True,
                 load_data = True,
                 delta = 0.4,
                 fps = 25):
        
        self.data_root = data_root
        self.loc_id = loc_id
        self.output_steps = output_steps
        self._train = train

        # Time between the last frame of the input and the output frame
        self.eps = 1.e-6
        self.delta = delta
        self.fps = fps
        self.skip_steps = int(self.delta * self.fps + self.eps)
        """ We use 2.4 seconds (60 frames) as a time window for the input and output.
            If delta = 0.4 seconds, we use 10 frames (0.4 * 25) as the time gap between the each time step (frame).
            Then, we use frames 1, 11, 21, 31, 41, and 51 as the input frames, and frame 61 as the output frame.
        """
        self.time_window = 61
        # Number of frames we should consider for the output
        self.output_frames = int(self.delta * self.fps + self.eps)

        self.state_veh_ids = [PRECEDING_ID, FOLLOWING_ID,
                              LEFT_PRECEDING_ID, LEFT_ALONGSIDE_ID, LEFT_FOLLOWING_ID,
                              RIGHT_PRECEDING_ID, RIGHT_ALONGSIDE_ID, RIGHT_FOLLOWING_ID]   
        self.train_ratio = train_ratio
        if load_data:
            self.LB_upper, self.RB_upper, self.LB_lower, self.RB_lower = 0, 0, 0, 0
            self.lane_ids = None
            self.X_train, self.X_test, self.y_train, self.y_test, self.num_features = self.preprocess_data()
        else:
            self.X_train, self.X_test, self.y_train, self.y_test, self.num_features = [], [], [], [], 0



    # Load and split training and test data
    def preprocess_data(self):
        # We read all data at the same location
        self.data = []
        self.lane_ids = None
        track_ids = []

        # We select 10 tracks for location 1
        nums = [11, 25, 30, 35, 45, 50, 57] if self.loc_id == 1 else range(60)
        #nums = [11] if self.loc_id == 1 else range(60)
        
        for di in nums:
            di = di if self.loc_id == 1 else di + 1
            data_dir = os.path.join(self.data_root, str(di).zfill(2) + '_recordingMeta.csv')
            if not os.path.exists(data_dir):
                continue
            info = pd.read_csv(data_dir)
            if info['locationId'][0] == self.loc_id:
                print(f'Process location {self.loc_id} | recording {di}')
                track_ids.append(di)
                # Load track data
                file_path = os.path.join(self.data_root, str(di).zfill(2) + '_tracks.csv')
                data, lane_ids = read_track_csv(file_path)
                self.data.append(data)
                # The lane ids are the same for all data at the same location
                lane_ids.sort()
                if self.lane_ids is None:
                    self.lane_ids = lane_ids
                else:
                    # We only consider the same lane ids (intersection set)
                    self.lane_ids = list(set(self.lane_ids).intersection(set(lane_ids)))
                    # Absurd case: vehicles on the emergent lane
                    if self.lane_ids != lane_ids:
                        print(f'Absurd lane ids: {lane_ids}, we only consider intersection: {self.lane_ids}')
                    
        if self.loc_id == 6:
            self.lane_num = 4 # 4 upper roads and 3 lower roads
        else:
            self.lane_num = int(len(self.lane_ids) / 2)
        num_vehicle = sum(len(data) for data in self.data)
        print(f'Number of vehicles: {num_vehicle}, lane ids: {self.lane_ids}')
        print('*'*50)
        

        # Construct the training and test set
        X_train, X_test, y_train, y_test = [], [], [], []
        
        # Process tracks one by one
        for track_i in range(len(self.data)):
            print(f'Process track {track_ids[track_i]}')
            data_tracks = self.data[track_i]
            
            # Calculate the road bounds
            self.LB_upper, self.RB_upper, self.LB_lower, self.RB_lower = self.Road_Bound(data_tracks)
            print(f'Upper road bounds | left: {self.LB_upper}, right: {self.RB_upper}')
            print(f'Lower road bounds | left: {self.LB_lower}, right: {self.RB_lower}')
            print('*'*50)
            
            # Process the trajectory of each vehicle (data_fs)
            for data_fs in data_tracks:
                X_veh = []
                X_all, y_all = [], []

                ''' For each frame, select features and prediction goals '''
                for f in data_fs:
                    # Filter out the vehicles that are not in the road bounds
                    lane_id = data_fs[f][LANE_ID]
                    if lane_id not in self.lane_ids:
                        continue
                    
                    # State for each frame
                    X_tmp = self.frame2state(data_fs[f], f, track_i)
                    
                    X_veh.append(X_tmp)
                
                    # X and y pair
                    if len(X_veh) == self.time_window:
                        X_state = X_veh[0::self.skip_steps]
                        X_state = np.array(X_state[:-1])
                        # y: The longitudinal and lateral acceleration of the ego vehicle in the next steps
                        y_state = np.array(X_veh[-1])
                        # Longitudinal and lateral acceleration of the ego vehicle
                        acc_lon = y_state[self.lane_num+3].reshape(-1, 1)
                        acc_lat = y_state[self.lane_num+5].reshape(-1, 1)
                        # Concatenate the acceleration of the ego vehicle
                        y = np.concatenate([acc_lon, acc_lat], axis = 1) # 1 * 2
                        
                        X_all.append(X_state)
                        y_all.append(y)
        
                        X_veh = X_veh[1:]
            
                # Split train and test set for each vehicle
                train_len = int(len(X_all) * self.train_ratio)
                X_train.extend(X_all[:train_len])
                y_train.extend(y_all[:train_len])
                X_test.extend(X_all[train_len:])
                y_test.extend(y_all[train_len:])

        return X_train, X_test, y_train, y_test, len(X_tmp)

    



    # For each direction, we calculate the left and right bounds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RB_upper
    #              <---  <---
    # --------------------------------------
    #              <---  <---
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LB_upper
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LB_lower
    #              --->  --->
    # --------------------------------------
    #              --->  --->
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RB_lower
    '''
        We use the position bounds of the vehicles to determine the road
    '''
    def Road_Bound(self, data_tracks, shift = 0.5):
        LB_upper, RB_upper = -100, 100
        LB_lower, RB_lower = 100, -100
            
        for data_fs in data_tracks:
            for f in data_fs:
                lane_id = data_fs[f][LANE_ID]
                try:
                    lane_index = self.lane_ids.index(lane_id)
                    x, y = data_fs[f][X], data_fs[f][Y] # Position of the vehicle (upper left corner)
                    height = data_fs[f][HEIGHT]
                    if lane_index < self.lane_num: # Upper road
                        LB_upper = max(LB_upper, y+ height)
                        RB_upper = min(RB_upper, y)
                    else: # Lower road
                        LB_lower = min(LB_lower, y)
                        RB_lower = max(RB_lower, y + height)
                except:
                    print('*'*50)
                    print(f'lane id {lane_id} is not in the lane ids {self.lane_ids}, we overlook this vehicle')
                    print(f'ego vehicle ID: {data_fs[f][TRACK_ID]}, frame: {f}')
        RB_upper -= shift
        LB_upper += shift
        LB_lower -= shift
        RB_lower += shift

        return LB_upper, RB_upper, LB_lower, RB_lower




    # identify the state of the ego and surrdound vehicles at each frame
    def frame2state(self, frame, fi, track_i):
        # X (input)
        #   (1) Lane id of the ego vehicle; 
        #   (2) Vehicle type, longitudinal and lateral velocity and acceleration, and distance to the road two sides
        #   (3) Vehicle type, longitudinal velocity and acceleration, and relative positions of surronding vehicles (proceeding, following, left, and right vehicles)
        X_tmp = np.zeros(self.lane_num + 8 + len(self.state_veh_ids)*5, dtype = np.float32)

        ''' Lane id of the ego vehicle '''
        lane_one_hot, lane_idx = self.Encode_Lane(frame[LANE_ID])
        SGN = -1 if lane_idx < self.lane_num else 1
        X_tmp[:self.lane_num] = lane_one_hot


        ''' The state of the ego vehicle '''
        veh_state = np.zeros(8, dtype = np.float32)
        # Vehicle type
        if frame[WIDTH] > 8: # truck
            veh_state[1] = 1.
        else:
            veh_state[0] = 1.
        # Longitudinal velocity and acceleration
        veh_state[2] = frame[X_VELOCITY] * SGN
        veh_state[3] = frame[X_ACCELERATION] * SGN
        # Lateral velocity and acceleration
        veh_state[4] = frame[Y_VELOCITY] * SGN
        veh_state[5] = frame[Y_ACCELERATION] * SGN
        # Distance to the road two sides
        ''' We build a local coordinate system for the ego vehicle,
            with the speed direction as the x-axis and the right-hand side direction as the y-axis,
            which determins the positive or negative sign of the lateral distance and relative positions.'''
        if lane_idx < self.lane_num:
            veh_state[6] = frame[Y] - self.RB_upper
            veh_state[7] = frame[Y] + frame[HEIGHT] - self.LB_upper
        else:
            veh_state[6] = self.RB_lower - frame[Y] - frame[HEIGHT]
            veh_state[7] = self.LB_lower - frame[Y]
        
        assert veh_state[6] > 0 and veh_state[7] < 0 # The vehicle is in the road bounds
        
        X_tmp[self.lane_num:self.lane_num+8] = veh_state


        ''' The state of the surronding vehicles '''
        for srd_i, srd_name in enumerate(self.state_veh_ids):
            veh_state = np.zeros(5, dtype = np.float32)
            id = frame[srd_name]
            # The surronding vehicle exists
            if id > 0:
                VALID = True
                try:
                    veh_frame = self.data[track_i][id-1][fi]
                except:
                    print('*'*50)
                    print(f'No frame {fi} for vehicle {id}')
                    continue
                # vehicle type
                if veh_frame[WIDTH] > 8: # truck
                    veh_state[0] = 1.
                else:
                    veh_state[1] = 1.
                # Longitudinal velocity and acceleration
                veh_state[2] = veh_frame[X_VELOCITY] * SGN
                veh_state[3] = veh_frame[X_ACCELERATION] * SGN
                # The relative position of the surronding vehicle
                if srd_name == RIGHT_ALONGSIDE_ID or srd_name == LEFT_ALONGSIDE_ID:
                    # Lateral distance for the alongside vehicles
                    veh_state[4] = (veh_frame[Y] - frame[Y] - veh_frame[HEIGHT]) * SGN
                
                # Longitudinal distance for the proceeding vehicles
                elif srd_name == RIGHT_PRECEDING_ID or srd_name == LEFT_PRECEDING_ID or srd_name == PRECEDING_ID:
                    if lane_idx < self.lane_num: # Upper road
                        veh_state[4] = (veh_frame[X] + veh_frame[WIDTH] - frame[X]) * SGN
                    else: # Lower road
                        veh_state[4] = (veh_frame[X] - frame[X] - frame[WIDTH]) * SGN
                    # Not valid case
                    if veh_state[4] < 0:
                        print('*'*50)
                        print(f'Preceding vehicles should be in the front of the ego vehicle, but {veh_state[4]} meters behind')
                        print(f'ego vehicle ID: {frame[TRACK_ID]}, frame: {fi}')
                        VALID = False
                # Longitudinal distance for the following vehicles
                else:
                    if lane_idx < self.lane_num: # Upper road
                        veh_state[4] = (veh_frame[X] - frame[X] - frame[WIDTH]) * SGN
                    else: # Lower road
                        veh_state[4] = (veh_frame[X] + veh_frame[WIDTH] - frame[X]) * SGN
                    # Not valid case
                    if veh_state[4] > 0:
                        print('*'*50)
                        print(f'Following vehicles should be in the back of the ego vehicle, but {veh_state[4]} meters ahead')
                        print(f'ego vehicle ID: {frame[TRACK_ID]}, frame: {fi}')
                        VALID = False
                
                # We only consider the valid case
                if not VALID:
                    veh_state = np.zeros(5, dtype = np.float32)       
        
            X_tmp[self.lane_num + 8 + srd_i*5 : self.lane_num + 8 + (1 + srd_i)*5] = veh_state

        return X_tmp


    


    # Encode the ego vehicle's lane id using one-hot encoder
    '''
        The encoded lane id is the same for the bi-directional lanes.
        For each direction, the lane id is encoded from the left to the right.
    '''
    def Encode_Lane(self, lane_id):
        lane_one_hot = np.zeros(self.lane_num)
        
        idx = self.lane_ids.index(lane_id)
        if idx < self.lane_num:
            lane_one_hot[self.lane_num - 1 - idx] = 1.
        else:
            lane_one_hot[idx - self.lane_num] = 1.

        return lane_one_hot, idx




    def __len__(self):
        if self._train:
            return len(self.X_train)
        else:
            return len(self.X_test)



    def __getitem__(self, idx):
        if self._train:
            return torch.from_numpy(self.X_train[idx]).float(), torch.from_numpy(self.y_train[idx]).float()
        else:
            return torch.from_numpy(self.X_test[idx]).float(), torch.from_numpy(self.y_test[idx]).float()