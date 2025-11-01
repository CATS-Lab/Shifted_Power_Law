import numpy as np


# We sample acceleration for each vehicle from predicted power law distribution.
class SampleAcc:
    def __init__(self,
                 loc_id = 4,
                 step_time = 0.2,
                 DISTRIBUTION = 'power_law',
                 input_steps = 50,
                 pred_steps = 1):
        
        # We only consider locations 2, 4, and 5 for the simulation to reduce the computation time
        assert loc_id in [4, 5], "Location ID must be one of [4, 5]"
        assert DISTRIBUTION in ['power_law', 'normal'], "Distribution must be either 'power_law' or 'normal'"
        
        # Location 4
        self.a_lon_loc4 = {0.2: 0.710}
        self.k_lon_loc4 = {0.2: -0.464}
        self.a_lat_loc4 = {0.2: 1.810}
        self.k_lat_loc4 = {0.2: -0.311}
        # Location 5
        self.a_lon_loc5 = {0.2: 3.411}
        self.k_lon_loc5 = {0.2: -0.211}
        self.a_lat_loc5 = {0.2: 1.410}
        self.k_lat_loc5 = {0.2: -0.347}
        # Collect all locations
        # Note: We only consider locations 4 and 5 for the simulation to reduce the computation time
        self.a_lon_locs = {4: self.a_lon_loc4, 5: self.a_lon_loc5}
        self.k_lon_locs = {4: self.k_lon_loc4, 5: self.k_lon_loc5}
        self.a_lat_locs = {4: self.a_lat_loc4, 5: self.a_lat_loc5}
        self.k_lat_locs = {4: self.k_lat_loc4, 5: self.k_lat_loc5}
        
        # A and k for each location with step time
        self.a_lon = self.a_lon_locs[loc_id][step_time]
        self.k_lon = self.k_lon_locs[loc_id][step_time]
        self.a_lat = self.a_lat_locs[loc_id][step_time]
        self.k_lat = self.k_lat_locs[loc_id][step_time]

        # Distribution type: power law or normal
        self.dist = DISTRIBUTION
        # Number of input steps used for predicting the acceleration
        self.num_input_steps = input_steps
        # Number of prediction steps used for controlling the acceleration
        self.num_pred_steps = pred_steps


    # Sample acceleration for each vehicle
    def sample_acc(self, vehs, pre_mu, pre_sigma):
        i = 0
        for veh in vehs:
            # We only consider the vehicles that have enough previous acceleration data and have not predicted acceleration
            if len(veh.prev_acc) < self.num_input_steps or len(veh.pred_acc_multisteps) > 0:
                continue
            # Predicted mu and sigma (we only consider the next frame)
            mu_lon, mu_lat = pre_mu[i, :self.num_pred_steps, 0], pre_mu[i, :self.num_pred_steps, 1]
            sigma_lon, sigma_lat = pre_sigma[i, :self.num_pred_steps, 0], pre_sigma[i, :self.num_pred_steps, 1]
            
            # Sample acceleration according to the power law
            if self.dist == 'power_law':
                acc_lons = self.sam_acc_power_law(self.a_lon, self.k_lon, mu_lon, sigma_lon)
                acc_lats = self.sam_acc_power_law(self.a_lat, self.k_lat, mu_lat, sigma_lat)
            # Sample acceleration according to the normal distribution
            else:
                acc_lons = self.sam_acc_normal(mu_lon, sigma_lon)
                acc_lats = self.sam_acc_normal(mu_lat, sigma_lat)
            
            # Update vehicle's acceleration for the next frame
            veh.pred_acc_multisteps = [[acc_lons[i], acc_lats[i]] for i in range(self.num_pred_steps)]
            veh.pred_acc = veh.pred_acc_multisteps.pop(0)

            i += 1
            
    
    # Sample acceleration based on inverse function of power law distribution
    def sam_acc_power_law(self, a, k, mus, sigmas):
        accs = []
        '''
            Original power law distribution: f(x) = -1 / (2*a*k*sigma) * (np.abs(x - mu) / a / sigma + 1) ** (1/k - 1)
            Inverse function: y = mu + a*sigma*np.sign(x - 0.5) * (np.abs(2 * x - 1)**k - 1)
        '''
        for mu, sigma in zip(mus, sigmas):
            x_inverse = np.random.uniform()
            y_inverse = mu + a*sigma*np.sign(x_inverse - 0.5) * (np.abs(2 * x_inverse - 1)**k - 1)
            accs.append(y_inverse)

        return accs
    

    # Sample acceleration based on normal distribution
    def sam_acc_normal(self, mus, sigmas):
        accs = []
        
        # Check if mus and sigmas are lists or numpy arrays
        if not isinstance(mus, (list, np.ndarray)):
            mus = [mus]
        if not isinstance(sigmas, (list, np.ndarray)):
            sigmas = [sigmas]
        
        for mu, sigma in zip(mus, sigmas):
            acc = np.random.normal(mu, sigma)
            accs.append(acc)
        
        return accs