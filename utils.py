import numpy as np
from skimage.measure import compare_ssim
import os

folder_name = 'real'
def get_train_directory(args):
    """
    Parameters
    ----------
    args :  args.data_opt--dataset to be used in training&testing
    Note: users should set the directories prior to running train file
    Returns
    -------
    directories of the kspace, sensitivity maps and mask

    """
    
    if args.data_opt == 't2shuffling':
        
        #kspace_dir = '../SSDU-all/t2shuffling/' + folder_name +'/Kspace.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/sunpply/Real/cmap_op1.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/sunpply/Real/rebuttal_cmap1.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/gt_exp/cmap_real_gt_.mat'
        kspace_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs4/Kspace.mat'
        coil_dir = '/home/molin/SSDU-Cmap/simulate/Cmap-gt.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs2/cmap_acs2_sim.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky_noise/cmap_sim_gt_noise-3.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/ablation_cmap/cmap_acs2_R16.mat'
        #kspace_dir = '/home/molin/SSDU-Cmap/New_data/Kspace.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/New_data/Cmap1.mat'        
        #coil_dir = '../SSDU-all/t2shuffling/'+folder_name+'/Cmap.mat'

    elif args.data_opt == 'Coronal_PDFS':

        kspace_dir = '...'
        coil_dir = '...'

    else:
        raise ValueError('Invalid data option')

    mask_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs2/Mask.mat'
    basis_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs4/Basis.mat'
    #mask_dir = '/home/molin/SSDU-Cmap/ablation_mask/mask_acs6_R16.mat'
    #mask_dir = '/home/molin/SSDU-Cmap/sunpply/Real/real_mask_2_r20_2.mat'
    #basis_dir = '../SSDU-all/t2shuffling/'+folder_name+'/Basis.mat'
    #mask_dir = '/home/molin/SSDU-Cmap/New_data/Mask.mat'
    #basis_dir = '/home/molin/SSDU-Cmap/New_data/Basis.mat'

    if os.path.exists('../SSDU-all/t2shuffling/'+folder_name+'/Init.mat'):
        coeffsvd_dir = '../SSDU-all/t2shuffling/'+folder_name+'/Init.mat'
    else:
        coeffsvd_dir = None
    if os.path.exists('/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs4/GT1.mat'):
        gtimg_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs4/GT.mat'
    else:
        gtimg_dir = None


    #pre_gt_dir = '/home/molin/SSDU-Cmap/sunpply/Real/rebuttal_gt1.mat'
    #pre_gt_dir = '/home/molin/SSDU-Cmap/ablation_imag/image_acs2_R16.mat'
    #pre_gt_dir = '/home/molin/SSDU-Cmap/Imag-gt.mat'
    #cmap_mask_dir = '/home/molin/SSDU-Cmap/gt_exp/cmap_mask_real_gt_.mat'
    pre_gt_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs2/gt-acs2.mat'
    #pre_gt_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky_noise/acs2/1gt_imag_noacs.mat'
    #cmap_mask_dir = '/home/molin/SSDU-Cmap/ablation_cmap1/cmap_acs2_R16_mask.mat'
    cmap_mask_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs2/cmap_acs2_mask_sim.mat'
    #cmap_mask_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky_noise/cmap_sim_gt_mask_noise-3.mat'
    #pre_gt_dir = '/home/molin/SSDU-Cmap/simulate/pre_gt.mat'

    print('\n kspace dir : ', kspace_dir, '\n \n coil dir :', coil_dir, '\n \n mask dir: ', mask_dir)

    return kspace_dir, coil_dir, mask_dir, basis_dir, coeffsvd_dir, gtimg_dir, pre_gt_dir, cmap_mask_dir


def get_test_directory(args):
    """
    Parameters
    ----------
    args :  args.data_opt--dataset to be used in training&testing
    Note: users should set the directories prior to running test file
    Returns
    -------
    directories of the kspace, sensitivity maps and mask and ;
    saved_model_dir : saved training model directory

    """
    if args.data_opt == 't2shuffling': 
        #kspace_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky_noise/acs4/Kspace.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky_noise/acs2/cmap_acs2_sim_noise.mat'
        #kspace_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs4/Kspace.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/cmap_espirit_sim.mat'
        #kspace_dir = '../SSDU-all/t2shuffling/' + folder_name +'/Kspace.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/gt_exp/cmap_real_gt_.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/sunpply/Real/rebuttal_cmap1.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/ablation_cmap/cmap_acs2_R16.mat'
        kspace_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs4/Kspace.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs2/cmap_acs2_sim.mat'        
        coil_dir = '/home/molin/SSDU-Cmap/simulate/Cmap-gt.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/sunpply/Real/cmap_op1.mat'
        #kspace_dir = '/home/molin/SSDU-Cmap/New_data/Kspace.mat'
        #coil_dir = '/home/molin/SSDU-Cmap/New_data/Cmap1.mat' 
        #coil_dir = '../SSDU-all/t2shuffling/'+folder_name+'/Cmap_r4_acs6line.mat'
        #coil_dir = '../SSDU-all/t2shuffling/'+folder_name+'/Cmap.mat'
        saved_model_dir = '/unborn/molin/4basis/15-02-2023-18-52-324basis_gt_10_lr_schedule_True_cmaprecon_False_use_imagrecon_True_imagres_5cmapres__rho_0.44fac_80TE_200Epochsinit_cmap+0.02init_cmap+2.0init_imag+0.0520cgimag__60cgcmap_lr_0.000510Unrolls_restore_False_new_guassianSelection'

    elif args.data_opt == 'Coronal_PDFS':

        kspace_dir = '...'
        coil_dir = '...'
        saved_model_dir = '...'

    else:
        raise ValueError('Invalid data option')
    mask_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs2/Mask.mat'
    basis_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs4/Basis.mat'
    #mask_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs2/Mask.mat'
    #basis_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs4/Basis.mat'
    #mask_dir = '/home/molin/SSDU-Cmap/sunpply/Real/real_mask_2_r20_2.mat'
    #mask_dir = '/home/molin/SSDU-Cmap/ablation_mask/mask_acs6_R16.mat'
    #mask_dir = '../SSDU-all/t2shuffling/'+folder_name+'/Mask_R8_acs6line.mat'
    #basis_dir = '../SSDU-all/t2shuffling/'+folder_name+'/Basis.mat'
    #mask_dir = '/home/molin/SSDU-Cmap/New_data/Mask.mat'
    #basis_dir = '/home/molin/SSDU-Cmap/New_data/Basis.mat'
    if os.path.exists('../SSDU-all/t2shuffling/'+folder_name+'/Init.mat'):
        coeffsvd_dir = '../SSDU-all/t2shuffling/'+folder_name+'/Init.mat'
    else:
        coeffsvd_dir = None
    if os.path.exists('/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs4/GT.mat'):
        gtimg_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs4/GT.mat'
    else:
        gtimg_dir = None

    #pre_gt_dir = '/home/molin/SSDU-Cmap/sunpply/Real/rebuttal_gt1.mat'
    #pre_gt_dir = '/home/molin/SSDU-Cmap/ablation_imag/image_acs6_R16.mat'
    pre_gt_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs2/gt-acs2.mat'
    #pre_gt_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky_noise/acs2/1gt_imag_noacs.mat'
    #pre_gt_dir = '/home/molin/SSDU-Cmap/Imag-gt.mat'
    #cmap_mask_dir = '/home/molin/SSDU-Cmap/ablation_cmap1/cmap_acs2_R16_mask.mat'
    #cmap_mask_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/cmap_espirit_mask_sim.mat'
    #cmap_mask_dir = '/home/molin/SSDU-Cmap/gt_exp/cmap_mask_real_gt_.mat'
    cmap_mask_dir = '/home/molin/SSDU-Cmap/simulate/echo80_8ky/acs2/cmap_acs2_mask_sim.mat'
    print('\n kspace dir : ', kspace_dir, '\n \n coil dir :', coil_dir,
          '\n \n mask dir: ', mask_dir, '\n \n saved model dir: ', saved_model_dir)

    return kspace_dir, coil_dir, mask_dir, basis_dir, coeffsvd_dir, gtimg_dir, saved_model_dir, pre_gt_dir, cmap_mask_dir


def getSSIM(space_ref, space_rec):
    """
    Measures SSIM between the reference and the reconstructed images
    """

    space_ref = np.squeeze(space_ref)
    space_rec = np.squeeze(space_rec)
    space_ref = space_ref / np.amax(np.abs(space_ref))
    space_rec = space_rec / np.amax(np.abs(space_ref))
    data_range = np.amax(np.abs(space_ref)) - np.amin(np.abs(space_ref))

    return compare_ssim(space_rec, space_ref, data_range=data_range,
                        gaussian_weights=True,
                        use_sample_covariance=False)


def getPSNR(ref, recon):
    """
    Measures PSNR between the reference and the reconstructed images
    """

    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    psnr = 20 * np.log10(np.abs(ref.max()) / (np.sqrt(mse) + 1e-10))

    return psnr


def fft(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : coil images of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform image space to k-space.

    """

    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ispace, axes=axes), axes=axes, norm=norm), axes=axes)
    
    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * kspace.shape[axis]

        kspace = kspace / np.sqrt(fact)

    return kspace


def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : image space of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform k-space to image space.

    """

    ispace = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * ispace.shape[axis]

        ispace = ispace * np.sqrt(fact)

    return ispace


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm .

    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations

    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


def F_hMy(input_kspace, cmap_kspace, cmap_ref, axes=(0, 1)):
    """
    Parameters
    ----------
    input_kspace : nrow x ncol x ncoil
    sens_maps : nrow x ncol x ncoil

    axes : The default is (0,1).

    Returns
    -------
    sense1 image

    """

    image_space = ifft(input_kspace, axes=axes, norm=None, unitary_opt=True) # W, H, C
    cmap_k = ifft(cmap_kspace, axes=axes, norm=None, unitary_opt=True) # 32, 12, 208, 256
    """
    To get a rough cmap with RSS
    """
    #Magnitude_img = abs(image_space)
    #Magnitude_img = np.sqrt(np.sum(Magnitude_img**2, axis = 2, keepdims = True))
    #cmap_est = image_space / Magnitude_img
    #cmap_est = cmap_ref
    """
    Use cmap_est to get init img
    """

    #Eh_op = np.conj(cmap_est) * image_space
    #Eh_op_cmap = np.conj(image_ref) * image_space
    #cmap_est = np.sum(Eh_op_cmap, axis=axes[-1] + 1, keepdims = False)
    cmap_est = cmap_ref
    Eh_op_imag = np.conj(cmap_ref) * image_space
    img_est = np.sum(Eh_op_imag, axis=axes[-1] + 1, keepdims = False)
    
    #print('origin',img_est.shape)
    #image_space = np.transpose(image_space, axis = [2, 0, 1])
    #cmap_k = np.transpose(cmap_k, axis = [2, 0, 1])
    return image_space, cmap_est, img_est, cmap_k






def complex2real(input_data):
    """
    Parameters
    ----------
    input_data : row x col
    dtype :The default is np.float32.

    Returns
    -------
    output : row x col x 2

    """

    return np.stack((input_data.real, input_data.imag), axis=-1)


def real2complex(input_data):
    """
    Parameters
    ----------
    input_data : row x col x 2

    Returns
    -------
    output : row x col

    """

    return input_data[..., 0] + 1j * input_data[..., 1]
