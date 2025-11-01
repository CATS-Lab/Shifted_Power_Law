# Description: This file contains the Simulation class, which is responsible for running the simulation of the traffic scenario.
class Simulation:
    def __init__(self, road, vehicles, dt = 1/25):
        self.road = road
        self.vehs = vehicles
        # Time step for each frame
        self.dt = dt


    # Run the simulation for one time step
    def run_step(self, vehs = None):
        vehs = vehs if vehs is not None else self.vehs
        for veh in vehs:
            # The vehicle has no next state but the predicted acceleration, we update the vehicle's position according to predicted acceleration
            if len(veh.next_loc) == 0 and veh.pred_acc is not None:
                veh.update_position(self.dt)
            # The vehicle has no next state and no predicted acceleration, we mark the vehicle as out of the road 
            elif len(veh.next_loc) == 0 and veh.pred_acc is None:
                veh.out_of_road = True
            # If the vehicle has a next state (loaded from dataset), update the vehicle's state
            else:
                veh.move_to_next_state()
                