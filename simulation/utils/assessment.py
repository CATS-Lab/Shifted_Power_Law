# This class is responsible for assessing the state of the vehicles and the road at each time step of the simulation.
class Assessment:
    def __init__(self, road, vehicles = None):
        self.road = road
        self.vehs = vehicles
        self.total_VKT = 0
        self.total_VKT_initial = 0


    # Check (1) whether carsh happens, (2) whether vehicles run out of the road
    def evaluate(self, vehs, id2veh):
        collision = False
        vehs_on_road = []
        new_id2veh = {}
        for veh in vehs:
            if veh.out_of_road:
                # Update the total VKT once the vehicle runs out of the road
                self.total_VKT += veh.VKT
                veh.VKT_counted = True
                continue
            
            # Filter out the vehicles that run out of the road
            if veh.loc[0] > self.road.length or veh.loc[0] < 0 or veh.loc[1] + veh.width > self.road.width*self.road.num_lanes or veh.loc[1] < 0:
                veh.out_of_road = True
                # Update the total VKT once the vehicle runs out of the road
                self.total_VKT += veh.VKT
                self.total_VKT_initial += veh.VKT_initial
                veh.VKT_counted = True
            else:
                veh.out_of_road = False
                vehs_on_road.append(veh)
                new_id2veh[veh.id] = veh
            
            # Check whether crash happens. We only need to check the distance between the current vehicle and the surrounding vehicles
            ''' [PRECEDING_ID, FOLLOWING_ID, LEFT_PRECEDING_ID, LEFT_ALONGSIDE_ID, LEFT_FOLLOWING_ID, RIGHT_PRECEDING_ID, RIGHT_ALONGSIDE_ID, RIGHT_FOLLOWING_ID] '''
            for srd_id in veh.srd_vehs:
                if srd_id >= 0: # We annotate -1 for the vehicles that are not in the surrounding area
                    srd_veh = id2veh[srd_id]
                    if self.check_collision(veh, srd_veh):
                        collision = True
                        veh.collision = True
                        srd_veh.collision = True
                        break
        
        return collision, vehs_on_road, new_id2veh
    


    # Determine whether two vehicles collide
    def check_collision(self, veh1, veh2):
        x1, y1 = veh1.loc
        x1_right, y1_right = x1 + veh1.length, y1 + veh1.width
        x2, y2 = veh2.loc
        x2_right, y2_right = x2 + veh2.length, y2 + veh2.width
        
        # Check whether two rectangles intersect
        if x1_right < x2 or x1 > x2_right or y1_right < y2 or y1 > y2_right:
            return False
        else:
            return True