import pandas as pd
import random

from infra.vehicle import Vehicle
from .DataLaundry import Cal_Road_Boundary



'''
    HighD dataset (six different locations): https://levelxdata.com/highd-dataset/
    Attributes: frame, id, x, y, width, height, xVelocity, yVelocity, xAcceleration, yAcceleration, frontSightDistance, backSightDistance
                dhw, thw, ttc, precedingXVelocity, precedingId, followingId, leftPrecedingId, leftAlongsideId, leftFollowingId
                rightPrecedingId, rightAlongsideId, rightFollowingId, laneId

'''


# This class is responsible for loading the state of the vehicles from the highD data file
class DataLoader:
    def __init__(self,
                 data_path,
                 step_time = 0.2,
                 step_frames = 5,
                 input_steps = 12,
                 input_frames = 51,
                 fps = 25):
        self.data_path = data_path

        # current frame id
        self.cur_frame_id = 0
        self.ini_frame_id = 0
        self.max_frame_id = 0
        
        # Input time horizon and step time
        self.fps = fps  # frames per second
        self.step_time = step_time  # seconds
        self.step_frames = step_frames  # number of frames in a step
        self.input_steps = input_steps  # number of input steps
        self.input_frames = input_frames # Number of input frames for the model

        # Road boundary calculated from the data
        self.LB, self.RB = 0, 0

        # Record vehicles on roads
        self.vehs_on_road = []
        # Link vehicles' id to the vehicle objects
        self.id2veh = {}



    # Initialize data
    def Initialize(self):
        # Load the data file
        self.data = pd.read_csv(self.data_path)
        self.data =self.data[self.data['xVelocity'] > 0.] # We only consider vehicles moving in the positive x-direction (lower roads)
        
        # Calculate the road boundary according to vehicles' trajectories
        self.LB, self.RB = Cal_Road_Boundary(self.data)

        


    # Load the initial state of vehicles from the data file
    def load_init_vehs(self, ini_frame_id = None, max_frame_id = None):
        self.id2veh = {}

        # The initial state of the vehicles (frame > 50 since we need the previous states for prediction)
        if max_frame_id is None:
            self.max_frame_id = self.data['frame'].max()
        else:
            self.max_frame_id = max_frame_id
        self.ini_frame_id = ini_frame_id if ini_frame_id is not None else random.randint(self.input_frames, self.max_frame_id-self.input_frames)
        # Update the current frame id
        self.cur_frame_id = self.ini_frame_id
        initial_state = self.data[self.data['frame'] == self.ini_frame_id]
        
        # Create vehicle objects record the initial state
        vehicles = self.state2vehs(initial_state)
        ''' 
            We first simulate vehicles' movements for 50 frames (2 seconds) according to their recorded trajectories in the data file.
            Then, we predict vehicles' acclerations for the next 25 frames (1 second) using previous 50-frame state.
            If a vehicle runs out of the roads, we remove it from the simulation.
        '''
        # Update the vehicels' future states for 50 frames (2 seconds)
        vehicles = self.load_future_states(vehicles)
        
        # Update vehicles on roads
        self.vehs_on_road = vehicles
        



    # Update vehicles (new entry vehicles and vehicles on roads)
    def update_vehs(self, entry_vehs):
        self.cur_frame_id += self.step_frames
        # Update vehicles on roads
        self.vehs_on_road.extend(entry_vehs)

        


    # Get new entry vehicles at the current frame
    def load_entry_vehs(self):
        # vehicle ids in the current frame
        cur_state = self.data[self.data['frame'] == self.cur_frame_id]
        cur_ids = cur_state['id'].unique()
        # vehicle ids in the previous frame
        prev_state = self.data[self.data['frame'] == self.cur_frame_id - self.step_frames]
        prev_ids = prev_state['id'].unique()
        # new entry vehicles
        entry_ids = [id for id in cur_ids if id not in self.id2veh.keys()]
        entry_state = cur_state[cur_state['id'].isin(entry_ids)]
        # Create vehicle objects for the new entry vehicles
        entry_vehs = self.state2vehs(entry_state)
        # Update the vehicels' future states for time horizon
        entry_vehs = self.load_future_states(entry_vehs)

        return entry_vehs


    # Convert the state to vehicle objects
    def state2vehs(self, state):
        vehicles = []
        for index, row in state.iterrows():
            veh_type = 'Car' if row['width'] < 8. else 'Truck'
            veh_height = 1.6 if veh_type == 'Car' else 3.5
            veh_length = 5 if veh_type == 'Car' else 12.
            veh_width = 2. if veh_type == 'Car' else 2.5
            veh = Vehicle(id = row['id'],
                        type = veh_type,
                        width = veh_width,
                        height = veh_height,
                        length = veh_length,
                        acc = [row['xAcceleration'], row['yAcceleration']],
                        speed = [row['xVelocity'], row['yVelocity']],
                        loc = [row['x'], row['y']])
            
            vehicles.append(veh)
            
            self.id2veh[row['id']] = veh

        return vehicles


    # Update the vehicels' future states for 50 frames (2 seconds)
    def load_future_states(self, vehs):
        # Get the next 50-frame states of the vehicles
        next_states = self.data[(self.data['frame'] > self.cur_frame_id) & (self.data['frame'] < self.cur_frame_id + self.input_frames)]
        for veh in vehs:
            # Continuous states of the vehicle (25 fps)
            veh.next_acc = next_states[next_states['id'] == veh.id][['xAcceleration', 'yAcceleration']].values.tolist()
            veh.next_speed = next_states[next_states['id'] == veh.id][['xVelocity', 'yVelocity']].values.tolist()
            veh.next_loc = next_states[next_states['id'] == veh.id][['x', 'y']].values.tolist()
            # We downsample the next states to match the input time
            veh.next_acc = veh.next_acc[self.step_frames-1::self.step_frames]
            veh.next_speed = veh.next_speed[self.step_frames-1::self.step_frames]
            veh.next_loc = veh.next_loc[self.step_frames-1::self.step_frames]

            # The next states plus the current state should be the same length as the input steps
            assert len(veh.next_acc) + 1 <= self.input_steps, f"Vehicles' next acceleration length mismatch: {len(veh.next_acc) + 1} > {self.input_steps}"
            
        return vehs
    

    