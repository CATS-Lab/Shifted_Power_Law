import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import torch.optim.lr_scheduler as LR_scheduler
import argparse


from dataset.highD import highD
from model.model import ego_acc_LSTM_dist
from model.criterion import dist_loss



# Parameters
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_id', type=int, default=4, help='location index')
    parser.add_argument('--delta', type=float, default=0.2, help='Time between the last frame of the input and the output frame')
    
    return parser.parse_args()




def main(model_type = 'LSTM'):
    args = parse_args()
    loc_i = args.loc_id

    #log file
    logger = logging.getLogger('')
    root = 'exp'
    os.makedirs(root, exist_ok=True)
    root = os.path.join(root, f'exp_highD')
    os.makedirs(root, exist_ok=True)
    # new log filefold
    result_file = os.path.join(root, f'highD_loc{args.loc_id}_{args.delta}s')
    os.makedirs(result_file, exist_ok=True)
    # set the log file path
    filehandler = logging.FileHandler(os.path.join(result_file, f'training_{loc_i}.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)


    # Set hyperparameters
    output_steps = 1
    batch_size = 128
    hidden_size = 128
    num_layers = 2
    num_epochs = 100
    lr = 0.0001
    
    logger.info('*'*50)
    logger.info(f'output_steps: {output_steps}, batch_size: {batch_size}, hidden_size: {hidden_size}, num_layers: {num_layers}, num_epochs: {num_epochs}, lr: {lr}')
    logger.info('*'*50)
    logger.info(f'delta: {args.delta}')
    logger.info('*'*50)
    # Load dataset
    data_root = '../data'
    
    logger.info(f'############## Process location {loc_i} ###############')
    # Train dataset
    train_dataset = highD(data_root = data_root,
                            loc_id = loc_i,
                            output_steps = output_steps,
                            train = True,
                            delta= args.delta)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Test dataset
    test_dataset = highD(data_root = data_root,
                            loc_id = loc_i,
                            output_steps = output_steps,
                            train = False,
                            load_data=False,
                            delta= args.delta)
    test_dataset.X_test = train_dataset.X_test
    test_dataset.y_test = train_dataset.y_test

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info('The number of data pairs for training: {}'.format(len(train_dataset)))
    logger.info('The number of data pairs for velidation: {}'.format(len(test_dataset)))
    
    logger.info('*'*50)
    # Lanes
    logger.info(f'Lane IDs: {train_dataset.lane_ids}')
    # Road bounds
    logger.info(f'Upper road bounds | left bound: {train_dataset.LB_upper}, right bound: {train_dataset.RB_upper}')
    logger.info(f'Lower road bounds | left bound: {train_dataset.LB_lower}, right bound: {train_dataset.RB_lower}')
    logger.info('*'*50)
    
    
    # Create model, loss function, and optimizer
    num_feature = train_dataset.num_features
    if model_type == 'LSTM_dist':
        model = ego_acc_LSTM_dist(num_feature = num_feature, hidden_size = hidden_size, num_layers = num_layers, output_size = output_steps)
        criterion = dist_loss()
    else:
        raise NotImplementedError
    model = model.to(device='cuda')
    
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Set up the learning rate scheduler
    lr_scheduler = LR_scheduler.MultiStepLR(optimizer, milestones=[60,90], gamma=0.1)

    best_mape = 9999

    # Train model
    logger.info('############# Start Training #############')
    for epoch in range(num_epochs):
        # current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        # logger.info(f'############# Starting Epoch {epoch+1} | LR: {current_lr} #############')
        
        train_loss = 0
        #train_loader = tqdm(train_loader, dynamic_ncols=True)
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device='cuda')
            y_batch = y_batch.to(device='cuda')

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        # Update learning rate
        lr_scheduler.step()


        if (epoch+1) % 1 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.6f}')
        

            # Evaluation
            test_lon_MAE = 0
            test_lat_MAE = 0
            test_MAE = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device='cuda')
                    y_batch = y_batch.to(device='cuda')
                    y_pred = model(X_batch)
                    
                    if model_type == 'LSTM_dist':
                        y_pred = y_pred[0] # Mean value
                    
                    loss = torch.mean(torch.abs(y_pred - y_batch).view(-1, 2), axis = 0) # (B, 2)
                    test_lon_MAE += loss[0].item()
                    test_lat_MAE += loss[1].item()
                    test_MAE += torch.mean(loss).item()

            MAE = test_MAE/len(test_loader)
            test_lat_MAE = test_lat_MAE/len(test_loader)
            test_lon_MAE = test_lon_MAE/len(test_loader)
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], longitudinal MAE: {test_lon_MAE:.4f}, lateral MAE: {test_lat_MAE:.4f}, MAE: {MAE:.4f}')
            
            if MAE < best_mape:
                best_mape = MAE
                # save the best model
                torch.save(model.state_dict(), f'{result_file}/best_{loc_i}.pth')

    # save the final model
    torch.save(model.state_dict(), f'{result_file}/final_{loc_i}.pth')



if __name__ == '__main__':
    main(model_type='LSTM_dist')