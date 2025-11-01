# Define a vehicle
class Vehicle:
    def __init__(self, id, type, width, height, length, acc, speed, loc):
        self.id = id
        self.type = type # Car or Truck
        self.width = width
        self.height = height
        self.length = length
        # longitudinal, lateral
        self.acc = acc
        self.speed = speed
        self.loc = loc
        
        # Lane id
        self.lane_id = None
        # previous lane id (50 frames)
        self.prev_lane_id = []


        # Surrounding vehicles
        ''' 
            [PRECEDING_ID, FOLLOWING_ID,
            LEFT_PRECEDING_ID, LEFT_ALONGSIDE_ID, LEFT_FOLLOWING_ID,
            RIGHT_PRECEDING_ID, RIGHT_ALONGSIDE_ID, RIGHT_FOLLOWING_ID]
        '''
        self.srd_vehs = []
        # previous surrounding vehicles (50 frames)
        self.prev_srd_vehs = []

        # Previous vehicle state (50 frames)
        self.prev_acc = []
        self.prev_speed = []
        self.prev_loc = []

        # Next vehicle state
        ''' Previous state may shorter than 2 seconds required for prediction, we use the next state to fill the gap '''
        self.next_acc = []
        self.next_speed = []
        self.next_loc = []

        # predicted acceleration
        self.pred_acc_multisteps = [] # Multisteps prediction
        self.pred_acc = None

        # Out of the road
        self.out_of_road = False

        # Collision
        self.collision = False

        # Vehicle kilometers traveled
        self.VKT = 0
        self.VKT_initial = 0
        self.VKT_counted = False


    # Update the vehicle's position and speed
    def update_position(self, dt):
        self.speed[0], x2 = self.cal_loc(self.acc[0], self.pred_acc[0], self.speed[0], self.loc[0], dt)
        self.speed[1], self.loc[1] = self.cal_loc(self.acc[1], self.pred_acc[1], self.speed[1], self.loc[1], dt)
        
        # Updata the acceleration
        self.acc = self.pred_acc    
        self.pred_acc = self.pred_acc_multisteps.pop(0) if len(self.pred_acc_multisteps) > 0 else None

        # Update the vehicle's traveled kilometers
        self.VKT += x2 - self.loc[0]
        
        self.loc[0] = x2
    

    # Calculate acc, speed, and loc for the next frame
    # v(t) = v_1 + a_1*t + 0.5*(a_2-a_1)/t_0 * t^2
    # x(t) = x_1 + v_1*t + 0.5*a_1*t^2 + (a_2-a_1)/6/t_0 * t^3
    def cal_loc(self, a_1, a_2, v_1, x_1, t0):
        v_2 = v_1 + a_1 * t0 + 0.5 * (a_2 - a_1) * t0
        x_2 = x_1 + v_1 * t0 + 0.5 * a_1 * t0 ** 2 + (a_2 - a_1) / 6 * t0 ** 2
        
        return v_2, x_2


    # Move to the next state (loaded from dataset)
    def move_to_next_state(self):
        assert len(self.next_acc) > 0 and len(self.next_speed) > 0 and len(self.next_loc) > 0, \
            f'Next state is empty for vehicle {self.id}. Please check the dataset.'
        
        # Update the vehicle's traveled kilometers in the initial state
        self.VKT_initial += self.next_loc[0][0] - self.loc[0]
        
        # Move to the next state
        self.loc = self.next_loc.pop(0)
        self.speed = self.next_speed.pop(0)
        self.acc = self.next_acc.pop(0)
        
        