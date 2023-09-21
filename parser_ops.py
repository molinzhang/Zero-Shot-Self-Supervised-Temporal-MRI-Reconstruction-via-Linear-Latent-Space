import argparse

from numpy import False_


def get_parser():
    parser = argparse.ArgumentParser(description='SSDU: Self-Supervision via Data Undersampling ')

    # %% hyperparameters for the unrolled network
    parser.add_argument('--acc_rate', type=int, default=4,
                        help='acceleration rate')
    parser.add_argument('--TE', type=int, default =80,
                        help='Number of TE images')
    parser.add_argument('--epochs', type=int, default=200, # 300
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-4, 
                        help='learning rate') #ismrm: 1e-3, 5e-4
    parser.add_argument('--lr_schedule', type=str, default=True, 
                        help='learning rate') #ismrm: 1e-3, 5e-4
    parser.add_argument('--batchSize', type=int, default=80,
                        help='batch size')
    parser.add_argument('--nb_unroll_blocks', type=int, default= 10, # 20
                        help='number of unrolled blocks')
    parser.add_argument('--nb_res_blocks_imag', type=int, default=10, # 10 
                        help="number of residual blocks in ResNet")
    parser.add_argument('--nb_res_blocks_cmap', type=int, default=5, # 10 
                        help="number of residual blocks in ResNet")
    parser.add_argument('--CG_Iter_imag', type=int, default=20,  #20
                        help='number of Conjugate Gradient iterations for DC')
    parser.add_argument('--CG_Iter_cmap', type=int, default=60,  #20
                        help='number of Conjugate Gradient iterations for DC')
    parser.add_argument('--use_new_kspace', type=str, default= False,
                        help='Generate gt kspace from gt image and mask')
    parser.add_argument('--multi_channel', type=str, default= False,
                        help='use multi channel method')
    parser.add_argument('--subspace', type=str, default= True,
                        help='use subspace method')
    parser.add_argument('--gradient_method', type=str, default= 'MG',
                        help='gradient method for cg block', choices = ['AG', 'MG'])
    parser.add_argument('--use_cmaprecon', type=str, default= False,
                        help='use cmap reconstruction')
    parser.add_argument('--use_imagrecon', type=str, default= True,
                        help='use imag reconstruction')                        
    parser.add_argument('--net_cfactor', type=int, default=4,
                        help='channel in image network')
    parser.add_argument('--use_cmapk', type=str, default= False,
                        help='use square ksapce for cmap reconstruction')
    parser.add_argument('--cmap_k', type=int, default=256,
                        help='size of square kspace for cmap reconstruction')
    parser.add_argument('--net_cmap', type=int, default=12,
                        help='channel in cmap network')
    parser.add_argument('--filters_imag', type=int, default=64,
                        help='filters in network')
    parser.add_argument('--filters_cmap', type=int, default=32,
                        help='filters in network')
    parser.add_argument('--init_para_cmap1', type=float, default=0.02,   # 0.02
                        help='filters in network')
    parser.add_argument('--init_para_cmap2', type=float, default=2.0,
                        help='filters in network')
    parser.add_argument('--init_para_imag', type=float, default=0.05,
                        help='filters in network')
    parser.add_argument('--order', type=str, default='denoise_first',
                        help='denoise block and cg block', choices=['denoise_first', 'cg_first'])
    parser.add_argument('--only_cg', type=str, default= False,
                        help='use cmap reconstruction')
    parser.add_argument('--restore', type=str, default= False,
                        help='use cmap reconstruction')
    parser.add_argument('--saved_model_dir', type=str, default= \
        #'/unborn/molin/16-02-2022-16-25-15_acs6R16_10use_cmaprecon_False_use_imagrecon_True_imagres_5cmapres_use_cmapk_False_rho_0.43fac_32TE_100Epochsinit_cmap+0.02init_cmap+2.0init_imag+0.0520cgimag__100cgcmap_lr_0.000510Unrolls_restore_False_new_guassianSelection',
        '/unborn/molin/rebuttal_mask1/08-04-2022-16-00-10_suprealsub__10_lr_schedule_True_cmaprecon_False_use_imagrecon_True_imagres_5cmapres__rho_0.43fac_32TE_200Epochsinit_cmap+0.01init_cmap+3.0init_imag+0.0520cgimag__60cgcmap_lr_0.000510Unrolls_restore_False_new_guassianSelection',
                        help='use cmap reconstruction')

    # %% hyperparameters for the dataset
    parser.add_argument('--data_opt', type=str, default='t2shuffling',
                        help=' directories for the kspace, sensitivity maps and mask')
    parser.add_argument('--nrow_GLOB', type=int, default=190,
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol_GLOB', type=int, default=256,
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--ncoil_GLOB', type=int, default=8,
                        help='number of coils of the slices in the dataset')

    # %% hyperparameters for the SSDU
    parser.add_argument('--mask_type', type=str, default='new_guassian',
                        help='mask selection for training and loss masks', choices=['Gaussian', 'Uniform','new_guassian'])
    parser.add_argument('--mask_up_freq', type=int, default=1,
                        help='update mask during training for multiple mask')
    parser.add_argument('--rho', type=float, default=0.4,
                        help='cardinality of the loss mask, \ rho = |\ Lambda| / |\ Omega|') # notice that the loss mask and DC mask are disjoint
    parser.add_argument('--multi_mask_test', type=int, default=1,
                        help='use multiple mask in testing') # notice that the loss mask and DC mask are disjoint

    # acslines ablation study
    #parser.add_argument('--acslines', type=int, default='4',
    #                    help='acslines', choices=[2, 4, 6, 8, 10, 12])
    #parser.add_argument('--acslines', type=int, default='4',
    #                    help='acslines', choices=[2, 4, 6, 8, 10, 12])



    # %% parameters for noise stat
    parser.add_argument('--std_n', type=float, default= 0.0000000,
                        help='std of guassian noise in kspace') 

    return parser
