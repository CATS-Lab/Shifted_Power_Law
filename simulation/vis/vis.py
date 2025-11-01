import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import cv2



# This class visualizes the state of the vehicles and the road at each time step of the simulation.
class Visualization:
    def __init__(self, road, vehicles = None):
        self.road = road
        self.vehicles = vehicles
        
        # Read figures of cars and trucks
        self.car_fig = plt.imread('vis/assets/car.jpg')
        self.truck_fig = plt.imread('vis/assets/truck.jpg')
        self.ego_car_fig = plt.imread('vis/assets/ego_car.jpg')



    def vis_state(self, vehicles = None, result_root = 'results', sim_id = 0, frame_id = 0):
        if vehicles is not None:
            self.vehicles = vehicles
        
        # Create the figure and axis
        aspect_ratio = self.road.length / self.road.width / self.road.num_lanes
        self.fig, self.ax = plt.subplots(figsize=(20, round(20/aspect_ratio*2, 3)), dpi=500)
        
        # Draw the road boundaries
        self.ax.add_patch(patches.Rectangle((0, 0), self.road.length, self.road.width*self.road.num_lanes,
                                           edgecolor = 'black', lw=2, facecolor='gray', fill=True, alpha=0.01))
        # Draw the lane lines at bottom
        for i in range(1, self.road.num_lanes):
            y = i * self.road.width
            self.ax.plot([0, self.road.length], [y, y], color='black', linestyle='--', linewidth=0.5)
            # Make sure the vehicles are not covered by lines
            self.ax.set_zorder(0)
        
        # Draw the vehicle as a rectangle
        for vehicle in self.vehicles:
            x, y = vehicle.loc
            length, width = vehicle.length, vehicle.width
            img_veh = self.car_fig if vehicle.type == 'Car' else self.truck_fig
            self.ax.imshow(img_veh,
                          extent=(x, x + length,
                                  y, y + width),
                          origin='upper', 
                          zorder=2)
            

        self.ax.set_xlim(1, self.road.length-1)
        self.ax.set_ylim(-1, self.road.width*self.road.num_lanes+1.)
        
        # Reverse the y-axis
        self.ax.invert_yaxis()

        # Save the figure
        os.makedirs(result_root, exist_ok=True)
        os.makedirs(os.path.join(result_root, f'vis_{sim_id}'), exist_ok=True)
        plt.savefig(os.path.join(result_root, f'vis_{sim_id}', f'frame_{frame_id}.jpg'), bbox_inches='tight')

        plt.close('all')




# Convert images to video
class image2video():
    def __init__(self, img_width, img_height):
        self.video_writer = None
        self.is_end = False
        self.img_width = img_width
        self.img_height = img_height 

    def start(self, file_name, fps):
        four_cc = cv2.VideoWriter_fourcc(*'mp4v')
        # four_cc = cv2.VideoWriter_fourcc(*'XVID')
        img_size = (self.img_width, self.img_height)

        self.video_writer = cv2.VideoWriter()
        self.video_writer.open(file_name, four_cc, fps, img_size, True)

    def record(self, img):
        if self.is_end is False:
            self.video_writer.write(img)

    def end(self):
        self.is_end = True
        self.video_writer.release()