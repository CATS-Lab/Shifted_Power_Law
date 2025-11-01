import os
import pandas as pd


'''
    Record the simulation process for visualization and analysis
'''
class Record:
    def __init__(self, results_root = 'results/', sim_id = 0):
        self.results_root = results_root
        self.sim_id = sim_id


    # Initialize the record
    def initialize(self):
        self.RECORD = pd.DataFrame(columns=['frame_id', 'id', 'type', 'width', 'height', 'length',
                                            'accX', 'accY', 'speedX', 'speedY', 'locX', 'locY',
                                            'PRECEDING_ID', 'FOLLOWING_ID', 'LEFT_PRECEDING_ID','LEFT_ALONGSIDE_ID', 'LEFT_FOLLOWING_ID',
                                            'RIGHT_PRECEDING_ID', 'RIGHT_ALONGSIDE_ID', 'RIGHT_FOLLOWING_ID',
                                            'land_id', 'out_of_road', 'collision'])
    
    # Record the current state of the vehicles
    def record_state(self, vehs, frame_id):
        for veh in vehs:
            self.RECORD = self.RECORD._append({'frame_id': frame_id, 'id': veh.id, 'type': veh.type, 'width': veh.width, 'height': veh.height, 'length': veh.length,
                                            'accX': veh.acc[0], 'accY': veh.acc[1], 'speedX': veh.speed[0], 'speedY': veh.speed[1], 'locX': veh.loc[0], 'locY': veh.loc[1],
                                            'PRECEDING_ID': veh.srd_vehs[0], 'FOLLOWING_ID': veh.srd_vehs[1], 'LEFT_PRECEDING_ID': veh.srd_vehs[2], 'LEFT_ALONGSIDE_ID': veh.srd_vehs[3], 'LEFT_FOLLOWING_ID': veh.srd_vehs[4],
                                            'RIGHT_PRECEDING_ID': veh.srd_vehs[5], 'RIGHT_ALONGSIDE_ID': veh.srd_vehs[6], 'RIGHT_FOLLOWING_ID': veh.srd_vehs[7],
                                            'land_id': veh.lane_id, 'out_of_road': veh.out_of_road, 'collision': veh.collision}, ignore_index=True)

    # Save the record
    def save(self):
        os.makedirs(self.results_root, exist_ok=True)
        self.RECORD.to_csv(os.path.join(self.results_root, f'record_{self.sim_id}.csv'))
    