################################################
#  Written by: Wang Chen
#  The University of Hong Kong
#  University of Wisconsin-Madison
#  wchen22@connnect.hku.hk
################################################


import os
import argparse
import logging
import torch
from tqdm import tqdm

from dataset.DataLoader import DataLoader
from infra.environment import Road
from vis.vis import Visualization
from vis.MakeVideo import make_video
from model.model import ego_acc_LSTM_dist
from utils.simulation import Simulation
from utils.frame2state import Frame2State
from utils.record import Record
from utils.assessment import Assessment
from utils.sample_acc import SampleAcc


# Define parameters
def parse_args():
    parser = argparse.ArgumentParser(description='Run the simulation')
    parser.add_argument('--loc_id', type=int, default=4, help='Location id of the highD data')
    parser.add_argument('--data_path', type=str, default='data/highD/07_tracks.csv', help='Path to the highD data')
    parser.add_argument('--res_root', type=str, default='results', help='Root path to the results')
    parser.add_argument('--vis', type=bool, default=False, help='Visualize the simulation results')
    parser.add_argument('--num_sim', type=int, default=1, help='Number of simulations')
    parser.add_argument('--dist', type=str, default='power_law', choices=["power_law", "normal"], help='Distribution type for sampling acceleration')
    parser.add_argument('--delta', type=float, default=0.2, help='Step time')
    parser.add_argument('--init_frame_id', type=int, default=None, help='Initial frame id for loading the vehicles. If None, it will be randomly selected.')
    parser.add_argument('--max_frame_id', type=int, default=None, help='Maximum frame id for loading the vehicles. If None, it will be set to the maximum frame id in the data.')
    
    return parser.parse_args()



def main():
    # Parse the arguments
    args = parse_args()
    # Create the results root
    res_root = args.res_root + '_' + args.dist
    os.makedirs(res_root, exist_ok=True)
    res_loc = os.path.join(res_root, f'results_loc{args.loc_id}_{args.delta}s')
    os.makedirs(res_loc, exist_ok=True)
    num_sim = len(os.listdir(res_loc))
    res_path = os.path.join(res_loc, f'sim_{num_sim}')
    os.makedirs(res_path, exist_ok=True)
    
    # Set the logger
    logger = logging.getLogger('')
    filehandler = logging.FileHandler(os.path.join(res_path, 'simulation.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    
    # Calculate input steps
    INPUT_TIME_HORIZON = 2.4 # seconds
    fps = 25 # frames per second
    epsilon = 1.e-3 # a small value to avoid numerical issues
    step_frames = int(args.delta * fps + epsilon) # Number of frames in a step
    input_steps = int(INPUT_TIME_HORIZON / args.delta + epsilon) # Number of input steps
    input_frames = int((input_steps - 1) * step_frames + epsilon) + 1 # Number of input frames for the model
    
    logger.info('## Start the simulation ##')
    logger.info(f'Location id: {args.loc_id}')
    logger.info(f'Number of simulations: {args.num_sim}')
    logger.info(f'Input time horizon: {INPUT_TIME_HORIZON} seconds')
    logger.info(f'Step time: {args.delta} seconds')
    logger.info(f'Step frames: {step_frames}')
    logger.info(f'Input steps: {input_steps}')
    logger.info(f'Input frames: {input_frames}')
    logger.info(f'Data path: {args.data_path}')
    logger.info("="*50)


    # Load the data
    DATA_LOADER = DataLoader(
        args.data_path,
        step_time=args.delta,
        step_frames = step_frames,
        input_steps = input_steps,
        input_frames = input_frames
        )
    logger.info(f'Loading the data from {args.data_path}')
    DATA_LOADER.Initialize()
    logger.info(f'Left boundary: {DATA_LOADER.LB}, Right boundary: {DATA_LOADER.RB}')
    
    # Create the ROAD (we only consider straight road)
    ROAD = Road(highD_id=args.loc_id, LB=DATA_LOADER.LB, RB=DATA_LOADER.RB)
    logger.info(f'Road width: {ROAD.width}, Number of lanes: {ROAD.num_lanes}')
    logger.info('#'*50)

    # Create the F2S for updating the state of the vehicles and converting the frame to the state for the model
    F2S = Frame2State(
        road_width=ROAD.width,
        num_lanes=ROAD.num_lanes,
        loc_id=args.loc_id,
        input_steps=input_steps
        )
    
    # Create the model
    logger.info(f'Loading the model from "./model/checkpoints/highD_loc{args.loc_id}_{args.delta}s.pth"')
    MODEL = ego_acc_LSTM_dist(num_feature = F2S.num_feature, hidden_size = 128, output_size = 1)
    MODEL.load_state_dict(torch.load(f'./model/checkpoints/highD_loc{args.loc_id}_{args.delta}s.pth'))
    MODEL.eval()
    MODEL.to('cuda')

    # Create the sample acceleration
    SAM_ACC = SampleAcc(
        loc_id = args.loc_id,
        step_time=args.delta,
        DISTRIBUTION=args.dist,
        input_steps=input_steps,
        pred_steps=1
        )
    
    # Record total simulated VKT
    total_sim_VKT = 0

    # Fot each dataset, we run the simulation for SIM_NUM times
    for sim_id in tqdm(range(args.num_sim), desc=f'Simulation Loc {args.loc_id}'):
        # Sample the initial vehicles from the data randomly
        DATA_LOADER.load_init_vehs(
            ini_frame_id=args.init_frame_id,
            max_frame_id=args.max_frame_id
        )
        logger.info(f'Initial frame: {DATA_LOADER.ini_frame_id}')

        # Coordinate the vehicles on the defined load
        ROAD.coord_vehs(DATA_LOADER.vehs_on_road)

        # Visuliaze
        if args.vis:
            VIS = Visualization(ROAD, DATA_LOADER.vehs_on_road)
            
        # Create the simulation
        SIM = Simulation(
            ROAD,
            DATA_LOADER.vehs_on_road,
            dt = args.delta
        )
        
        # Create the record for recording the sates of the vehicles
        RECORD = Record(results_root=res_path, sim_id=sim_id)
        RECORD.initialize()
        
        # Update and record the first frame
        F2S.update_surrd_vehs(DATA_LOADER.vehs_on_road)
        RECORD.record_state(DATA_LOADER.vehs_on_road, DATA_LOADER.ini_frame_id)
        
        # Create the assessment for evaluating the performance and safety of the vehicles
        ASSESS = Assessment(ROAD, DATA_LOADER.vehs_on_road)

        # Run the simulation
        for frame_id in tqdm(range(DATA_LOADER.ini_frame_id, DATA_LOADER.max_frame_id, step_frames), desc=f'Running Sim {sim_id}'):
            # Update the state of the ego vehicle and its surrounding vehicles
            F2S.update_surrd_vehs(DATA_LOADER.vehs_on_road)

            # Predict the acceleration of each vehicle
            if frame_id - DATA_LOADER.ini_frame_id + 1 >= input_frames:
                # Prepare the state of the ego vehicle and surrounding vehicles for the prediction model
                state = F2S.frame2state(DATA_LOADER.vehs_on_road, DATA_LOADER.id2veh)
                if len(state) > 0:
                    state = torch.tensor(state).float()
                    state = state.to('cuda')
                    # Predict the acceleration of the ego vehicle
                    veh_acc = MODEL(state) # [mu, sigma]
                    pred_mu = veh_acc[0].cpu().detach().numpy()    # N * 1 * 2
                    pred_sigma = veh_acc[1].cpu().detach().numpy() # N * 1 * 2
                    # Sample the acceleration of each vehicle
                    SAM_ACC.sample_acc(DATA_LOADER.vehs_on_road, pred_mu, pred_sigma)
            
            # Run the simulation for one time step
            SIM.run_step(DATA_LOADER.vehs_on_road)
            
            # Evaluate the performance and safety of the vehicles
            COLLISION, new_vehs_on_road, new_id2veh = ASSESS.evaluate(DATA_LOADER.vehs_on_road, DATA_LOADER.id2veh)
           
            # Record the state of the vehicles
            RECORD.record_state(DATA_LOADER.vehs_on_road, frame_id)

            if COLLISION:
                logger.info('*'*50)
                logger.info(f'!!! Collision happens at frame {frame_id}. Simulation ends !!!')
                logger.info('*'*50)
                # Save the record involving the collision
                RECORD.save()
                break
            
            # We filter out the vehicles that run out of the road to reduce the memory usage
            DATA_LOADER.vehs_on_road = new_vehs_on_road
            #DATA_LOADER.id2veh = new_id2veh

            # Load new entry vehicles
            new_vehs = DATA_LOADER.load_entry_vehs()
            ROAD.coord_vehs(new_vehs)
            # Update vehicles
            DATA_LOADER.update_vehs(new_vehs)
            

            # Visualize
            if args.vis:
                VIS.vis_state(DATA_LOADER.vehs_on_road, result_root = res_path, sim_id=sim_id, frame_id = frame_id + 1)
            
        # We only save the record if there is collision, otherwise, we only calculate the performance and safety of the vehicles
        for veh in DATA_LOADER.vehs_on_road:
            if not veh.VKT_counted:
                ASSESS.total_VKT += veh.VKT
                ASSESS.total_VKT_initial += veh.VKT_initial
                veh.VKT_counted = True
        
        logger.info(f'Total initial VKT: {ASSESS.total_VKT_initial / 1000:.2f} km')
        logger.info(f'Total simulated VKT: {ASSESS.total_VKT / 1000:.2f} km')
        total_VKT = (ASSESS.total_VKT + ASSESS.total_VKT_initial) / 1000
        logger.info(f'Total VKT: {total_VKT:.2f} km')
        logger.info('#'*10)

        total_sim_VKT += ASSESS.total_VKT

        # Make video
        if args.vis:
            root = os.path.join(res_path, f'vis_{sim_id}')
            make_video(root, DATA_LOADER.ini_frame_id, DATA_LOADER.max_frame_id, step_frames=step_frames)

       

    logger.info(f'Total simulated VKT: {total_sim_VKT / 1000:.2f} km')
    logger.info('## End the simulation ##')




if __name__ == '__main__':
    main()