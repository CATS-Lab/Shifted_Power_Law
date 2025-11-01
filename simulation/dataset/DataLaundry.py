# Calculate the boundary of the road according to vehicles' trajectories
# We use the upper and lower coordinates of all vehicles to calculate the boundary of the road.
def Cal_Road_Boundary(data):
    Shift = 0.5
    LB= data['y'].min() - Shift
    data['right'] = data['y'] + data['height']
    RB = data['right'].max() + Shift

    return LB, RB
    
    