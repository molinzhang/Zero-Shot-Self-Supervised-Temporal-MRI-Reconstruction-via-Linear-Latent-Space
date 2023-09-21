import numpy as np
import utils


class ssdu_masks():
    """

    Parameters
    ----------
    rho: split ratio for training and loss mask. \ rho = |\Lambda|/|\Omega|
    small_acs_block: keeps a small acs region fully-sampled for training masks
    if there is no acs region, the small acs block should be set to zero
    input_data: input k-space, nrow x ncol x ncoil
    input_mask: input mask, nrow x ncol

    Gaussian_selection:
    -divides acquired points into two disjoint sets based on Gaussian  distribution
    -Gaussian selection function has the parameter 'std_scale' for the standard deviation of the distribution. We recommend to keep it as 2<=std_scale<=4.

    Uniform_selection: divides acquired points into two disjoint sets based on uniform distribution

    Returns
    ----------
    trn_mask: used in data consistency units of the unrolled network
    loss_mask: used to define the loss in k-space

    """

    def __init__(self, rho=0.4, small_acs_block=(4, 4)):
        self.rho = rho
        self.small_acs_block = small_acs_block
    
    def new_guassian_selection(self, input_data, input_mask, std_scale = 4):
        
        nrow, ncol = input_data.shape[0], input_data.shape[1]
        if len(input_data.shape) == 2:
            axis_s = ()
        elif len(input_data.shape) == 3:
            axis_s = (2,)
        center_kx = int(utils.find_center_ind(input_data, axes=(1,) + axis_s))
        center_ky = int(utils.find_center_ind(input_data, axes=(0,) + axis_s))    
        
        guass_mask = np.zeros_like(input_mask)
        
        x_c = np.random.randn(int(nrow*ncol*self.rho)) * ((ncol - 1) /  std_scale) + center_kx
        y_c = np.random.randn(int(nrow*ncol*self.rho)) * ((nrow - 1) /  std_scale) + center_ky
        
        x_c = np.rint(x_c).astype(int)
        y_c = np.rint(y_c).astype(int)
        
        x_c[x_c > ncol -1] = ncol -1
        x_c[x_c < 0] = 0
        y_c[y_c > nrow -1] = nrow -1
        y_c[y_c < 0] = 0
                
        guass_mask[y_c, x_c] = 1

        guass_mask[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2:center_ky + self.small_acs_block[1] // 2] = 0
        
        loss_mask = input_mask * guass_mask
        
        #print(sum(sum(loss_mask)))
        
        #print(np.sum(loss_mask, (0,1)) / np.sum(input_mask, (0,1)))
        
        #print(f'\n Gaussian selection is processing, rho = {self.rho:.2f}, center of kspace: center-kx: {center_kx}, center-ky: {center_ky}')        
        
        return  input_mask - loss_mask, loss_mask
        
        
        
    def Gaussian_selection(self, input_data, input_mask, std_scale=4, num_iter=1):

        nrow, ncol = input_data.shape[0], input_data.shape[1]
        if len(input_data.shape) == 2:
            axis_s = ()
        elif len(input_data.shape) == 3:
            axis_s = (2,)
        center_kx = int(utils.find_center_ind(input_data, axes=(1,) + axis_s))
        center_ky = int(utils.find_center_ind(input_data, axes=(0,) + axis_s))    

        if num_iter == 0:
            print(f'\n Gaussian selection is processing, rho = {self.rho:.2f}, center of kspace: center-kx: {center_kx}, center-ky: {center_ky}')

        temp_mask = np.copy(input_mask)
        
        temp_mask[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2:center_ky + self.small_acs_block[1] // 2] = 0

        loss_mask = np.zeros_like(input_mask)
        count = 0

        while count <= np.int(np.ceil(np.sum(input_mask[:]) * self.rho)):

            indx = np.int(np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / std_scale)))
            indy = np.int(np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / std_scale)))
            
            
            if (0 <= indx < nrow and 0 <= indy < ncol and temp_mask[indx, indy] == 1 and loss_mask[indx, indy] != 1):
                loss_mask[indx, indy] = 1
                count = count + 1

        trn_mask = input_mask - loss_mask

        return trn_mask, loss_mask

    def uniform_selection(self, input_data, input_mask, num_iter=1):

        nrow, ncol = input_data.shape[0], input_data.shape[1]

        if len(input_data.shape) == 2:
            axis_s = ()
        elif len(input_data.shape) == 3:
            axis_s = (2,)
        center_kx = int(utils.find_center_ind(input_data, axes=(1,) + axis_s))
        center_ky = int(utils.find_center_ind(input_data, axes=(0,) + axis_s))    

        if num_iter == 0:
            print(f'\n Uniformly random selection is processing, rho = {self.rho:.2f}, center of kspace: center-kx: {center_kx}, center-ky: {center_ky}')

        temp_mask = np.copy(input_mask)
        temp_mask[center_kx - self.small_acs_block[0] // 2: center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2: center_ky + self.small_acs_block[1] // 2] = 0

        pr = np.ndarray.flatten(temp_mask)
        ind = np.random.choice(np.arange(nrow * ncol),
                               size=np.int(np.count_nonzero(pr) * self.rho), replace=False, p=pr / np.sum(pr))

        [ind_x, ind_y] = utils.index_flatten2nd(ind, (nrow, ncol))

        loss_mask = np.zeros_like(input_mask)
        loss_mask[ind_x, ind_y] = 1

        trn_mask = input_mask - loss_mask

        return trn_mask, loss_mask
