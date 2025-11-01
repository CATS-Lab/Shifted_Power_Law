import numpy as np


'''
    Given a frame, update the state of the each vehicle in the frame.
    Prepare the state of the ego vehicle and surrounding vehicles for the prediction model.
'''
class Frame2State:
    def __init__(self,
                 origin_x = 0,
                 origin_y = 0,
                 road_width = 3.5,
                 num_lanes = 3,
                 loc_id = 6,
                 input_steps = 50):
        
        self.ori_x = origin_x
        self.ori_y = origin_y
        self.road_width = road_width
        self.num_lanes = num_lanes
        self.loc_id = loc_id
        if loc_id == 6:
            self.num_lanes += 1
        
        # Number of input steps for the model
        self.input_steps = input_steps

        # Number of feature for the model
        self.num_feature = self.num_lanes + 8 + 8*5


    
    # We first need to identify the lane id based on the y-coordinate of the vehicle.
    # Then, we sort the vehicles on each lane according to x-coordinate of vehicles.
    def sort_vehs(self, vehs):
        vehs_on_lane = [[] for _ in range(self.num_lanes)]
        # Calculate the lane id for each vehicle
        for veh in vehs:
            # Update lane id
            veh.lane_id = self.get_lane_id(veh.loc[1] + veh.width / 2)
            veh.prev_lane_id.append(veh.lane_id)
            # We only record the previous self.input_steps frames for prediction
            if len(veh.prev_lane_id) > self.input_steps:
                veh.prev_lane_id.pop(0)

            # Initialize the surrounding vehicles
            veh.srd_vehs = [-1 for _ in range(8)]
            vehs_on_lane[veh.lane_id].append(veh)
            
            # Update the previous acc, speed, and loc
            veh.prev_acc.append(veh.acc)
            veh.prev_speed.append(veh.speed)
            veh.prev_loc.append(veh.loc)
            # We only record the previous self.input_steps frames for prediction
            if len(veh.prev_acc) > self.input_steps:
                veh.prev_acc.pop(0)
                veh.prev_speed.pop(0)
                veh.prev_loc.pop(0)
        
        # Sort the vehicles on the lane id according to their x-coordinate
        for i in range(self.num_lanes):
            vehs_on_lane[i].sort(key = lambda veh: veh.loc[0])
        
        return vehs_on_lane


    # Identify lane id based on the y-coordinate
    def get_lane_id(self, y):
        return int(y // self.road_width)
    


    # Identify 8 sorrounding vehicles of the given vehicle
    ''' 
        [PRECEDING_ID, FOLLOWING_ID,
        LEFT_PRECEDING_ID, LEFT_ALONGSIDE_ID, LEFT_FOLLOWING_ID,
        RIGHT_PRECEDING_ID, RIGHT_ALONGSIDE_ID, RIGHT_FOLLOWING_ID]
    '''
    def update_surrd_vehs(self, vehs):
        vehs_on_lane = self.sort_vehs(vehs)
        for lane_id, lane_vehs in enumerate(vehs_on_lane):
            for i, veh in enumerate(lane_vehs):
                # Preceeding and following vehicles on the same lane
                if i < len(lane_vehs) - 1:
                    veh.srd_vehs[0] = lane_vehs[i + 1].id # Preceeding vehicle
                if i > 0:
                    veh.srd_vehs[1] = lane_vehs[i - 1].id # Following vehicle

                # Right vehicles
                if lane_id < self.num_lanes - 1:
                    RIGHT = False
                    for j, veh2 in enumerate(vehs_on_lane[lane_id + 1]):
                        # The first vehicle on the right lane is in the front of the vehicle
                        if veh2.loc[0] + veh2.length >= veh.loc[0]:
                            # Right following vehicle
                            if j > 0:
                                veh.srd_vehs[7] = vehs_on_lane[lane_id + 1][j - 1].id
                            # Right alongside vehicle
                            if veh2.loc[0] <= veh.loc[0] + veh.length:
                                veh.srd_vehs[6] = veh2.id
                                # Right preceding vehicle
                                if j < len(vehs_on_lane[lane_id + 1]) - 1:
                                    veh.srd_vehs[5] = vehs_on_lane[lane_id + 1][j + 1].id
                            # Right preceding vehicle
                            else:
                                veh.srd_vehs[5] = veh2.id
                            
                            RIGHT = True
                            break
                    
                    # No right alongside or preceeding vehicle
                    if not RIGHT:
                        assert len(vehs_on_lane[lane_id + 1]) == 0 or vehs_on_lane[lane_id + 1][-1].loc[0] < veh.loc[0]
                        if len(vehs_on_lane[lane_id + 1]) > 0:
                            veh.srd_vehs[7] = vehs_on_lane[lane_id + 1][-1].id
                                    
                # Left vehicles
                if lane_id > 0:
                    LEFT = False
                    for j, veh2 in enumerate(vehs_on_lane[lane_id - 1]):
                        # The first vehicle on the left lane is in the front of the vehicle
                        if veh2.loc[0] + veh2.length >= veh.loc[0]:
                            # Left following vehicle
                            if j > 0:
                                veh.srd_vehs[4] = vehs_on_lane[lane_id - 1][j - 1].id
                            # Left alongside vehicle
                            if veh2.loc[0] <= veh.loc[0] + veh.length:
                                veh.srd_vehs[3] = veh2.id
                                # Left preceding vehicle
                                if j < len(vehs_on_lane[lane_id - 1]) - 1:
                                    veh.srd_vehs[2] = vehs_on_lane[lane_id - 1][j + 1].id
                            # Left preceding vehicle
                            else:
                                veh.srd_vehs[2] = veh2.id
                            
                            LEFT = True
                            break
                    
                    # No left alongside or preceeding vehicle
                    if not LEFT:
                        assert len(vehs_on_lane[lane_id - 1]) ==0 or vehs_on_lane[lane_id - 1][-1].loc[0] < veh.loc[0]
                        if len(vehs_on_lane[lane_id - 1]) > 0:
                            veh.srd_vehs[4] = vehs_on_lane[lane_id - 1][-1].id
                
                veh.prev_srd_vehs.append(veh.srd_vehs)
                # We only record the previous self.input_steps frames for prediction
                if len(veh.prev_srd_vehs) > self.input_steps:
                    veh.prev_srd_vehs.pop(0)
        
    


    # identify the state of the ego and surrdound vehicles at each frame
    def frame2state(self, vehs, id2veh):
        '''
            X (input)
                (1) Lane id of the ego vehicle; 
                (2) Vehicle type, longitudinal and lateral velocity and acceleration, and distance to the road two sides
                (3) Vehicle type, longitudinal velocity and acceleration, and relative positions of surronding vehicles (proceeding, following, left, and right vehicles)
        '''
        X_tmp = np.zeros((self.input_steps, self.num_feature), dtype = np.float32)
        X = []
        
        # Calculate the state for each vehicle
        for i, veh in enumerate(vehs):
            # We do not consider the vehicles that have no self.input_steps previous states or still have predicted acceleration
            if len(veh.prev_acc) < self.input_steps or len(veh.pred_acc_multisteps) > 0:
                continue
            
            ''' Lane id of the ego vehicle '''
            lane_one_hot = self.Encode_Lane(veh.prev_lane_id)
            X_tmp[:, :self.num_lanes] = lane_one_hot

            ''' The state of the ego vehicle '''
            # Vehicle type
            if veh.type == 'truck':
                X_tmp[:, self.num_lanes] = 1.
            else:
                X_tmp[:, self.num_lanes + 1] = 1.
            # Longitudinal velocity and acceleration
            X_tmp[:, self.num_lanes+2] = np.array(veh.prev_speed)[:, 0]
            X_tmp[:, self.num_lanes+3] = np.array(veh.prev_acc)[:, 0]
            # Lateral velocity and acceleration
            X_tmp[:, self.num_lanes+4] = np.array(veh.prev_speed)[:, 1]
            X_tmp[:, self.num_lanes+5] = np.array(veh.prev_acc)[:, 1]
            # Distance to the road two sides
            width = self.road_width * (self.num_lanes-1) if self.loc_id == 6 else self.road_width * self.num_lanes
            X_tmp[:, self.num_lanes+6] = width - np.array(veh.prev_loc)[:, 1] - veh.width # This is positive in local coordinates
            X_tmp[:, self.num_lanes+7] = self.ori_y - np.array(veh.prev_loc)[:, 1] # This is negative in local coordinates

            ''' The state of the surronding vehicles '''
            for srd_i in range(8):
                veh_state = np.zeros((self.input_steps, 5), dtype = np.float32)
                ids = np.array(veh.prev_srd_vehs)[:, srd_i]
                
                # Frame id for the surrounding vehicles
                fi_srd = 0
                
                # Calculate the state for each surrounding vehicle frame by frame
                for fi, srd_id in enumerate(ids):
                    if srd_id >= 0:
                        # Surrounding vehicle
                        srd_veh = id2veh[srd_id]
                        # Vehicle type
                        if srd_veh.type == 'truck':
                            veh_state[fi, 0] = 1.
                        else:
                            veh_state[fi, 1] = 1.
                        # Longitudinal velocity and acceleration
                        veh_state[fi, 2] = srd_veh.prev_speed[fi_srd][0]
                        veh_state[fi, 3] = srd_veh.prev_acc[fi_srd][0]
                        
                        # Lateral distance for the alongside vehicles
                        if srd_i == 3 or srd_i == 6:
                            veh_state[fi, 4] = srd_veh.prev_loc[fi_srd][1] - veh.prev_loc[fi][1] - srd_veh.width
                        # Longitudinal distance for the proceeding vehicles
                        elif srd_i == 0 or srd_i == 2 or srd_i == 5:
                            veh_state[fi, 4] = srd_veh.prev_loc[fi_srd][0] - veh.prev_loc[fi][0] - veh.length
                        # Longitudinal distance for the following vehicles
                        else:
                            veh_state[fi, 4] = srd_veh.prev_loc[fi_srd][0] + srd_veh.length - veh.prev_loc[fi][0]

                        fi_srd += 1
                   
                    '''If the surrounding vehicle in the current frame is not the same with the previous frame, we need to reset the frame id for the surrounding vehicle'''
                    if fi < self.input_steps - 1 and ids[fi] != ids[fi + 1]:
                        fi_srd = 0     
                
        
                X_tmp[:, self.num_lanes + 8 + srd_i*5 : self.num_lanes + 8 + (1 + srd_i)*5] = veh_state
            
            X.append(X_tmp)

        return np.array(X)


    


    # Encode the ego vehicle's lane id using one-hot encoder
    def Encode_Lane(self, lane_ids):
        lane_one_hot = np.zeros((len(lane_ids), self.num_lanes), dtype = np.float32)
        
        for i, lane_id in enumerate(lane_ids):
            lane_one_hot[i, lane_id] = 1.
        
        return lane_one_hot