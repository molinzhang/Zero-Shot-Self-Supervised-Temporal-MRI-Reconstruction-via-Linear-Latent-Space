import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py as h5
import time
import utils
import parser_ops
import UnrollNet
import masks.ssdu_masks as ssdu_masks

parser = parser_ops.get_parser()
args = parser.parse_args()
args.gradient_method = 'AG'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# .......................Load the Data...........................................
print('\n Loading ' + args.data_opt + ' test dataset...')
kspace_dir, coil_dir, mask_dir, basis_dir, coeffsvd_dir, gtimg_dir, saved_model_dir, pre_gt_dir, cmap_mask_dir = utils.get_test_directory(args)

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset
kspace_test = sio.loadmat(kspace_dir)['kspace']
sens_maps_testAll = sio.loadmat(coil_dir)['cmap']
#sens_maps_testAll = sio.loadmat('/home/molin/SSDU-Cmap/2cmap_reconfrom_imag_gt.mat')['cmap']
#sens_maps_testAll = sio.loadmat('Cmap_check.mat')['Cmap']
original_mask = sio.loadmat(mask_dir)['mask']
#original_mask[:] = 1
kspace_test = kspace_test[:args.TE]
sens_maps_testAll = sens_maps_testAll[:args.TE]
original_mask = original_mask[:args.TE]
basis = sio.loadmat(basis_dir)['basis'][:args.TE,:args.net_cfactor] # 10, 4
basis = np.reshape(basis, [args.TE, -1 ,1 ,1])

cmap_mask = sio.loadmat(cmap_mask_dir)
cmap_mask = cmap_mask['mask_cmap'][:args.TE] # 32, 208 ,256, 12

pre_gt = sio.loadmat(pre_gt_dir)
pre_gt = pre_gt['gt']

###
#pre_gt = np.zeros(np.shape(cmap_mask))
#pre_gt = pre_gt[:,:,:,0]
###
pre_gt = pre_gt[:args.TE]
pre_gt = utils.complex2real(pre_gt)



sens_maps_testAll = sens_maps_testAll * cmap_mask[:args.TE]

if coeffsvd_dir is not None:
    coeff_svd = sio.loadmat(coeffsvd_dir)['coeffs'] # 256, 256, 4
if gtimg_dir is not None:
    gt_image = sio.loadmat(gtimg_dir)['images'][:,:,:args.TE] # 256, 256, 10


if args.use_new_kspace:
    sens_img = np.expand_dims(np.transpose(gt_image, (2, 0, 1)), -1) * sens_maps_testAll # 10, 256, 256, 8
    np_kspace = np.empty(sens_img.shape, dtype=np.complex64)
    for slice_id in range(sens_img.shape[0]):
        np_kspace[slice_id, ...] = utils.fft(sens_img[slice_id, ...])
    np_kspace = np_kspace * np.expand_dims(original_mask, -1)

    noise_map = np.random.normal(0, args.std_n, size=np.shape(np_kspace) +(2,)).view(np.complex128)[...,0]
    np_kspace = np_kspace + noise_map

    kspace_test = np_kspace

else:
    kspace_check = kspace_test * (1 - np.expand_dims(original_mask, -1))
    np_kspace = kspace_test * np.expand_dims(original_mask, -1)

    kspace_test = np_kspace

print('\n Normalize kspace to 0-1 region')
#for ii in range(np.shape(kspace_test)[0]):
#    kspace_test[ii, :, :, :] = kspace_test[ii, :, :, :] / np.max(np.abs(kspace_test[ii, :, :, :][:]))

# %% Train and loss masks are kept same as original mask during inference
nSlices, *_ = kspace_test.shape
#test_mask = np.complex64(np.tile(original_mask[np.newaxis, :, :], (nSlices, 1, 1)))

ssdu_masker = ssdu_masks.ssdu_masks(args.rho)

test_mask = original_mask

print('\n size of kspace: ', kspace_test.shape, ', maps: ', sens_maps_testAll.shape, ', mask: ', test_mask.shape)

# %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency
if args.data_opt == 'Coronal_PD':
    test_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
    test_mask[:, :, 352:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

test_refAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)


print('\n generating the refs and sense1 input images')
test_inputAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
for ii in range(nSlices):
    sub_kspace = kspace_test[ii] * np.tile(test_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    #test_refAll[ii] = utils.sense1(kspace_test[ii, ...], sens_maps_testAll[ii, ...])
    #test_inputAll[ii] = utils.sense1(sub_kspace, sens_maps_testAll[ii, ...])

def gen_mask(sens_m):
    sens_m = utils.real2complex(sens_m)
    sens_m = np.transpose(sens_m, (0, 2, 3, 1))
    trn_mask, loss_mask, cmap_mask = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                      np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                        np.zeros((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

    nw_input = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
    cmap_input = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
    test_inputAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)

    init_cmap = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
    init_imag = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

    for ii in range(nSlices):
        #if np.mod(ii, 50) == 0:
        #    print('\n Iteration: ', ii)
        
        if args.mask_type == 'Gaussian':
            trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.Gaussian_selection(kspace_test[ii], original_mask[ii], num_iter=ii)

        elif args.mask_type == 'Uniform':
            trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.uniform_selection(kspace_test[ii], original_mask[ii], num_iter=ii)
    
        elif args.mask_type == 'new_guassian':
            trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.new_guassian_selection(kspace_test[ii], original_mask[ii])
        
        else:
            raise ValueError('Invalid mask selection')

        if args.multi_mask_test > 1:
            pass
        else:
            trn_mask[ii, ...] = original_mask[ii, ...]

        cmap_mask[ii, 104 - min(int(args.cmap_k // 2),0) : 104 + max(int(args.cmap_k // 2),104), 128 - int(args.cmap_k // 2) : 128 + int(args.cmap_k // 2)] = 1
        cmap_mask[ii, ...] = cmap_mask[ii, ...] * original_mask[ii, ...]
        cmap_kspace = kspace_test[ii] * np.tile(cmap_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))        
        sub_kspace = kspace_test[ii] * np.tile(trn_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        kspace_test[ii, ...] = kspace_test[ii] * np.tile(loss_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        nw_input[ii, ...], init_cmap[ii, ...], init_imag[ii, ...], cmap_input[ii, ...] = utils.F_hMy(sub_kspace, cmap_kspace, sens_m[ii, ...]) # recon the images with sub-mask of kspace

    ## average the cmap from nslice sub-kspace (RSS)
    init_cmap = np.sum(init_cmap, axis = 0, keepdims = True) 
    init_cmap = np.repeat(init_cmap, nSlices, axis = 0)
    init_cmap = np.transpose(init_cmap, (0, 3, 1, 2)) # 1, C, W, H


    #sio.savemat('maskss.mat', {'loss' : loss_mask, 'train': trn_mask})

    # %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations
    # for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
    # in the training mask we set corresponding columns as 1 to ensure data consistency
    if args.data_opt == 'Coronal_PD':
        trn_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
        trn_mask[:, :, 352:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

    # %% Prepare the data for the training
    
    #ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
    nw_input = utils.complex2real(np.transpose(nw_input, (0, 3, 1, 2)))
    cmap_input = utils.complex2real(np.transpose(cmap_input, (0, 3, 1, 2)))
    init_cmap = utils.complex2real(init_cmap)
    init_imag = utils.complex2real(init_imag)
    #print('\n size of ref kspace: ', ref_kspace.shape, ', nw_input: ', nw_input.shape, ', maps: ', sens_maps.shape, ', mask: ', trn_mask.shape)

    return 0, nw_input, trn_mask, loss_mask, init_cmap, init_imag, cmap_input, cmap_mask





sens_maps_testAll = np.transpose(sens_maps_testAll, (0, 3, 1, 2))
sens_maps_testAll = utils.complex2real(sens_maps_testAll)
cmap_mask = np.transpose(cmap_mask, (0, 3, 1, 2))

print('\n  loading the saved model ...')
tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sens_init = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='sens_maps_init')
img_init = tf.placeholder(tf.float32, shape=(None, None, None, 2), name='img_init')
pre_cmap = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='pre_cmap')
pre_imag = tf.placeholder(tf.float32, shape=(None, None, None, 2), name='pre_cmap')
trn_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='trn_mask')
loss_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='loss_mask')
nw_inputP = tf.placeholder(tf.float32, shape=(None, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB,  2), name='nw_input')
BasisP = tf.placeholder(tf.complex64, shape=(None, None, 1, 1), name='Basis')
Cmap_maskP = tf.placeholder(tf.float32, shape=(None, None, None, None), name='cmap_mask')
Cmap_kspaceP = tf.placeholder(tf.float32, shape=(None, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, 2), name='cmap_kspace')


# %% creating the dataset
#dataset = tf.data.Dataset.from_tensor_slices(( nw_inputP, sens_mapsP, trn_maskP, loss_maskP))
#dataset = dataset.shuffle(buffer_size=10 * args.batchSize)
#dataset = dataset.batch(args.batchSize)
#dataset = dataset.prefetch(args.batchSize)
#iterator = dataset.make_initializable_iterator()
#nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor = iterator.get_next('getNext')
#basis_tensor = tf.constant(basis, dtype = tf.complex64)

nw_output_img, nw_output_cmap, nw_output_kspace, INTER, MU1, MU2, MU3, imag_0, cmap_0 = UnrollNet.UnrolledNet(nw_inputP, Cmap_kspaceP, sens_init, img_init, trn_maskP, loss_maskP, BasisP, Cmap_maskP, pre_cmap, pre_imag, args.gradient_method).model

outs = []
with tf.Session(config=config) as sess:
    
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    
    sess.run(tf.global_variables_initializer())
    
    saver.restore(sess, tf.train.latest_checkpoint(saved_model_dir))
    
    # ..................................................................................................................
    graph = tf.get_default_graph()

    for iters in range(args.multi_mask_test):
        if args.multi_mask_test > 1:
            _, test_inputAll, trn_mask, loss_mask, init_cmap, init_imag, cmap_input, _ = gen_mask(sens_maps_testAll)
            test_mask = trn_mask

        _, test_inputAll, trn_mask, loss_mask, init_cmap, init_imag, cmap_input, _ = gen_mask(sens_maps_testAll)
        test_mask = trn_mask
        all_ref_slices, all_input_slices, all_recon_slices = [], [], []

        ITERs = args.TE // args.batchSize

        for ii in range(ITERs):
            batch_idx = range(ii * args.batchSize, (ii+1) * args.batchSize)
            #ref_image_test = np.copy(test_refAll[batch_idx, :, :])
            nw_input_test = np.copy(test_inputAll[batch_idx, :, :, :, :])
            cmap_input_test = np.copy(cmap_input[batch_idx, :, :, :, :])
            init_cmap = np.copy(init_cmap[batch_idx, :, :, :, :])
            init_imag = np.copy(init_imag[batch_idx, :, :, :])
            testMask = np.copy(test_mask[batch_idx, :, :])
            cmapmask = np.copy(cmap_mask[batch_idx, :, :,:])
            Basis_test = np.copy(basis[batch_idx, :, :,:])
            sens_maps_test = np.copy(sens_maps_testAll[batch_idx, :, :, :])
            pre_gt_test = np.copy(pre_gt[batch_idx, :, :, :])
            #ref_image_test, nw_input_test = utils.complex2real(ref_image_test), utils.complex2real(nw_input_test)   
      
            tic = time.time()
            dataDict = {nw_inputP: nw_input_test, trn_maskP: testMask, loss_maskP: testMask, \
                sens_init: init_cmap, img_init:init_imag, BasisP: Basis_test, Cmap_maskP: cmapmask,\
                   Cmap_kspaceP: cmap_input_test, pre_cmap: sens_maps_test, pre_imag: pre_gt}


            nw_output_ssdu, output_cmap= sess.run([nw_output_img,nw_output_cmap], feed_dict=dataDict) # 1, 4, 256, 256, 2
        
            toc = time.time() - tic
            #ref_image_test = utils.real2complex(ref_image_test.squeeze())
            nw_input_test = utils.real2complex(nw_input_test.squeeze())
            nw_output_ssdu = utils.real2complex(nw_output_ssdu)
            output_cmap = utils.real2complex(output_cmap)
            #sio.savemat("test_results_coeff.mat", {'test_img' : nw_output_ssdu})

            if args.subspace == True:
                nw_output_ssdu = np.sum(nw_output_ssdu * basis, axis = 1, keepdims = False)

            if args.data_opt == 'Coronal_PD':
                """window levelling in presence of fully-sampled data"""
                factor = np.max(np.abs(ref_image_test[:]))
            else:
                factor = 1

            #ref_image_test = np.abs(ref_image_test) / factor
            #nw_input_test = np.abs(nw_input_test) / factor
            #nw_output_ssdu = np.abs(nw_output_ssdu) / factor

            # ...............................................................................................................
            all_recon_slices.append(nw_output_ssdu)
            #all_ref_slices.append(ref_image_test)
            all_input_slices.append(nw_input_test)
            print('\n Iteration: ', ii, 'elapsed time %f seconds' % toc)
        
        outs.append(all_recon_slices)
        

print(len(all_recon_slices))
print(len(all_ref_slices))
print(len(all_input_slices))
if not os.path.exists(str(saved_model_dir).split('/')[-2]):
    os.makedirs(str(saved_model_dir).split('/')[-2])
print(str(saved_model_dir).split('/')[-2])    
sio.savemat(str(saved_model_dir).split('/')[-2] +'/'+ str(saved_model_dir).split('/')[-1] + ".mat", {'test_img' : np.array(outs), 'test_cmap':output_cmap})

