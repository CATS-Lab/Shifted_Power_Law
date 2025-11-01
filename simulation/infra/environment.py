''' Information of six highways in highD datasets '''
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
bounds = [
    [12.3, 25.5, 30.5, 44.0],
    [7.5, 16.5, 20.4, 29.3],
    [12.1, 24.8, 26.2, 39.0],
    [2.2, 14.8, 18.4, 31.8],
    [6.6, 16.1, 20.1, 29.3],
    [4.6, 19.7, 23.0, 36.1]
]
lanes_nums = [[3,3], [2,2], [3,3], [3,3], [2,2], [4,3]]


# road class
class Road:
    def __init__(self, length = 140, width = 3, num_lanes = 3, highD_id = None, LB = None, RB = None):
        if highD_id is not None:
            highD_id = highD_id - 1

            # We only consider the lower roads
            self.length = length
            self.num_lanes = lanes_nums[highD_id][1]
            if LB is not None and RB is not None:
                self.LB, self.RB = LB, RB
            else:
                self.LB, self.RB = bounds[highD_id][2], bounds[highD_id][3]
            self.width = (self.RB - self.LB) / self.num_lanes # for each lane
        else:
            self.length = length
            self.width = width # for each lane
            self.num_lanes = num_lanes
            self.LB, self.RB = 0, self.width * self.num_lanes


    # Coordinate vehicles' location on the road (redefined coordinate system)
    def coord_vehs(self, vehs):
        for veh in vehs:
            veh.loc[1] = veh.loc[1] - self.LB
            for i, loc in enumerate(veh.next_loc):
                veh.next_loc[i][1] = loc[1] - self.LB


    # Check vehicle's road id and distance to boundary
    def get_veh_loc(self, veh):
        x, y = veh.loc
        W = veh.width
        # land id
        lane_id = int(y / self.width)
        # distance to the left boundary (positive value)
        left_dist = y
        # distance to the right boundary (negative value)
        right_dist = y + W - self.width*self.num_lanes

        return lane_id, left_dist, right_dist
