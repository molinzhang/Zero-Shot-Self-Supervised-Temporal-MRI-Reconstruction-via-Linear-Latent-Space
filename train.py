import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import h5py as h5
import utils
import tf_utils
import parser_ops
import masks.ssdu_masks as ssdu_masks
import UnrollNet
from datetime import datetime

parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

save_dir ='/unborn/molin/4basis/'
Time_now = datetime.now()
Time_now = Time_now.strftime('%d-%m-%Y-%H-%M-%S')
pATH = str(args.saved_model_dir).split('/')[-1]
directory = os.path.join(save_dir, Time_now +'4basis_gt' +'_' + str(args.nb_res_blocks_imag) + '_lr_schedule_'+str(args.lr_schedule)+'_cmaprecon_'+str(args.use_cmaprecon)+'_use_imagrecon_'+str(args.use_imagrecon)\
                        +'_imagres_' + str(args.nb_res_blocks_cmap) +'cmapres_'+'_rho_'+str(args.rho)\
                        + str(args.net_cfactor) +'fac_' +str(args.TE)+'TE_'+str(args.epochs)+'Epochs' + 'init_cmap+' + str(args.init_para_cmap1) + 'init_cmap+' + str(args.init_para_cmap2)\
                        + 'init_imag+' + str(args.init_para_imag)+ str(args.CG_Iter_imag) +'cgimag_'+ '_' + str(args.CG_Iter_cmap) +'cgcmap_'+ 'lr_'+ str(args.learning_rate) + str(args.nb_unroll_blocks) + 'Unrolls_' \
                        +'restore_'+str(args.restore)+'_'+args.mask_type+'Selection')

if not os.path.exists(directory):
   os.makedirs(directory)

#print('\n create a test model for the testing')
#test_graph_generator = tf_utils.test_graph(directory)

#...............................................................................
start_time = time.time()
print('.................SSDU Training.....................')
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data..........................................
print('\n Loading ', args.data_opt, ' data, acc rate : ', args.acc_rate, ', mask type :', args.mask_type)
kspace_dir, coil_dir, mask_dir, basis_dir, coeffsvd_dir, gtimg_dir, pre_gt_dir, cmap_mask_dir = utils.get_train_directory(args)

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset
kspace_train = sio.loadmat(kspace_dir)['kspace'][:args.TE] # 10, 256, 256, 8
sens_maps = sio.loadmat(coil_dir)['cmap'][:args.TE] # 32, 256, 256, 12
#sens_maps = sio.loadmat('/home/molin/SSDU-Cmap/2cmap_reconfrom_imag_gt.mat')['cmap'][:args.TE]
original_mask = sio.loadmat(mask_dir)['mask'][:args.TE] # 10, 256, 256
#original_mask[:] = 1
basis = sio.loadmat(basis_dir)['basis'][:args.TE, :args.net_cfactor] # 10, 4
basis = np.reshape(basis, [args.TE, -1 ,1 ,1])
#coeff_svd = sio.loadmat(coeffsvd_dir)['coeffs'] # 256, 256, ?
#gt_image = sio.loadmat(gtimg_dir)['images'][:,:,:args.TE] # 256, 256, 10



cmap_mask = sio.loadmat(cmap_mask_dir)
cmap_mask = cmap_mask['mask_cmap'][:args.TE] # 32, 208 ,256, 12

sens_maps = sens_maps * cmap_mask[:args.TE]


pre_gt = sio.loadmat(pre_gt_dir)
pre_gt = pre_gt['gt'] 
####
#pre_gt = np.zeros(np.shape(cmap_mask))
#pre_gt = pre_gt[:,:,:,0]
####

pre_gt = utils.complex2real(pre_gt)


## Use gt image to generate gt kspace. Ksapce_train may not be correct...
if args.use_new_kspace:
    sens_img = np.expand_dims(np.transpose(gt_image, (2, 0, 1)), -1) * sens_maps # 10, 256, 256, 8
    np_kspace = np.empty(sens_img.shape, dtype=np.complex64)
    #check_img = np.empty((args.TE, 190, 256), dtype=np.complex64)
    
    for slice_id in range(sens_img.shape[0]):
        np_kspace[slice_id, ...] = utils.fft(sens_img[slice_id, ...])  # 10, 256, 256, 8
        #check_img[slice_id] = utils.ifft(utils.fft(gt_image[:,:, slice_id]))
    
    noise_map = np.random.normal(0, args.std_n, size=np.shape(np_kspace) +(2,)).view(np.complex128)[...,0]
    #noise_map_imag = np.random.normal(0, args.std_n, size=np.shape(np_kspace)[:3] +(1,))
    np_kspace = np_kspace + noise_map

    #kspace_check = np_kspace * (1 - np.expand_dims(original_mask, -1))
    #np_kspace = np_kspace 
    
    #kspace_train = np_kspace * np.expand_dims(original_mask, -1)
else:
    kspace_check = kspace_train * (1 - np.expand_dims(original_mask, -1))
    np_kspace = kspace_train * np.expand_dims(original_mask, -1)
    kspace_train = np_kspace


#sio.savemat('/home/molin/SSDU-Cmap/simulate/echo80_8ky_noise/acs4/Kspace.mat',{'kspace': np_kspace})

kspace_check = np.transpose(utils.complex2real(kspace_check),[0,3,1,2,4])


print('\n Normalize the kspace to 0-1 region')
#for ii in range(np.shape(kspace_train)[0]):
#    kspace_train[ii, :, :, :] = kspace_train[ii, :, :, :] / np.max(np.abs(kspace_train[ii, :, :, :][:]))

print('\n size of kspace: ', kspace_train.shape, ', maps: ', sens_maps.shape, ', mask: ', original_mask.shape)
nSlices, *_ = kspace_train.shape


print('\n create training and loss masks and generate network inputs... ')
ssdu_masker = ssdu_masks.ssdu_masks(args.rho)
sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))
sens_maps = utils.complex2real(sens_maps)

cmap_mask = np.transpose(cmap_mask, (0, 3, 1, 2))


def gen_mask(sens_m):
    sens_m = utils.real2complex(sens_m)
    sens_ms = np.transpose(sens_m, (0, 2, 3, 1))
    trn_mask, loss_mask, cmap_mask = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                      np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                        np.zeros((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.float32)

    nw_input = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
    cmap_input = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
    ref_kspace = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)

    init_cmap = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
    init_imag = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

    gt = sio.loadmat('GT_img.mat')
    gt = gt['img']
    for ii in range(nSlices):
        #if np.mod(ii, 50) == 0:
        #    print('\n Iteration: ', ii)
        
        if args.mask_type == 'Gaussian':
            trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.Gaussian_selection(kspace_train[ii], original_mask[ii], num_iter=ii)

        elif args.mask_type == 'Uniform':
            trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.uniform_selection(kspace_train[ii], original_mask[ii], num_iter=ii)
    
        elif args.mask_type == 'new_guassian':
            trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.new_guassian_selection(kspace_train[ii], original_mask[ii])
        
        else:
            raise ValueError('Invalid mask selection')
        
        cmap_mask[ii, 104 - min(int(args.cmap_k // 2),0) : 104 + max(int(args.cmap_k // 2),104), 128 - int(args.cmap_k // 2) : 128 + int(args.cmap_k // 2)] = 1
        cmap_mask[ii, ...] = cmap_mask[ii, ...] * original_mask[ii, ...]
        cmap_kspace = kspace_train[ii] * np.tile(cmap_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        sub_kspace = kspace_train[ii] * np.tile(trn_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        ref_kspace[ii, ...] = kspace_train[ii] * np.tile(loss_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        nw_input[ii, ...], init_cmap[ii, ...], init_imag[ii, ...], cmap_input[ii, ...] = utils.F_hMy(sub_kspace, cmap_kspace, sens_ms[ii, ...]) # recon the images with sub-mask of kspace
        #init_imag[ii,...] = gt[ii, ...]

    ## average the cmap from nslice sub-kspace (RSS)
    init_cmap = np.sum(init_cmap, axis = 0, keepdims = True) / nSlices
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
    
    ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
    nw_input = utils.complex2real(np.transpose(nw_input, (0, 3, 1, 2)))
    cmap_input = utils.complex2real(np.transpose(cmap_input, (0, 3, 1, 2)))
    init_cmap = utils.complex2real(init_cmap)
    init_imag = utils.complex2real(init_imag)
    #print('\n size of ref kspace: ', ref_kspace.shape, ', nw_input: ', nw_input.shape, ', maps: ', sens_maps.shape, ', mask: ', trn_mask.shape)

    return ref_kspace, nw_input, trn_mask, loss_mask, init_cmap, init_imag, cmap_input, cmap_mask

#ref_kspace, nw_input, trn_mask, loss_mask, init_cmap, init_imag, cmap_input, cmap_mask = gen_mask()
#print(ref_kspace.shape, nw_input.shape, trn_mask.shape, loss_mask.shape, init_cmap.shape, init_imag.shape, cmap_input.shape, cmap_mask.shape)
# %% set the batch size
total_batch = int(np.floor(np.float32(nSlices) / (args.batchSize)))
kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='refkspace')
kspaceCheck = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='ksapce_check')
sens_init = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='sens_maps_init')
pre_cmap = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='pre_cmap')
pre_imag = tf.placeholder(tf.float32, shape=(None, None, None, 2), name='pre_cmap')
img_init = tf.placeholder(tf.float32, shape=(None, None, None, 2), name='img_init')
trn_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='trn_mask')
loss_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='loss_mask')
Cmap_maskP = tf.placeholder(tf.float32, shape=(None, None, None, None), name='cmap_mask')
nw_inputP = tf.placeholder(tf.float32, shape=(None, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, 2), name='nw_input')
Cmap_kspaceP = tf.placeholder(tf.float32, shape=(None, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, 2), name='cmap_kspace')
BasisP = tf.placeholder(tf.complex64, shape=(None, None, 1, 1), name='Basis')
# %% creating the dataset
dataset = tf.data.Dataset.from_tensor_slices((kspaceP, nw_inputP, sens_init, img_init, trn_maskP, loss_maskP, kspaceCheck, BasisP, Cmap_kspaceP, Cmap_maskP, pre_cmap, pre_imag))
#dataset = dataset.shuffle(buffer_size=args.batchSize)
dataset = dataset.batch(args.batchSize)
dataset = dataset.prefetch(args.batchSize)
iterator = dataset.make_initializable_iterator() 
ref_kspace_tensor, nw_input_tensor, sens_init_tensor, imag_init_tensor, trn_mask_tensor, loss_mask_tensor, kspaceCheck_tensor, basis_tensor, Cmap_kspaceP_tensor, Cmap_maskP_tensor, pre_cmap_tensor, pre_imag_tensor = iterator.get_next('getNext')

# %% make training model
#basis_tensor = tf.constant(basis, dtype = tf.complex64)
nw_output_img, nw_output_cmap, nw_output_kspace, INTER, MU1, MU2, MU3, imag_0, cmap_0 = UnrollNet.UnrolledNet(nw_input_tensor, Cmap_kspaceP_tensor, sens_init_tensor, imag_init_tensor, trn_mask_tensor, loss_mask_tensor, basis_tensor, Cmap_maskP_tensor, pre_cmap_tensor, pre_imag_tensor, args.gradient_method).model
## every time sess.run([element]) is executed, the graph will generate a new element. 

#print('out name',nw_output_img.name)

scalar = tf.constant(0.5, dtype=tf.float32)
#weight = np.ones((args.TE, 12, 208, 256, 2))
#weight[0:5,:,:,:,:] = 0
#Weight = tf.constant(weight, dtype=tf.float32)

loss = tf.multiply(scalar, tf.norm((ref_kspace_tensor - nw_output_kspace)) / tf.norm(ref_kspace_tensor)) + \
tf.multiply(scalar, tf.norm((ref_kspace_tensor - nw_output_kspace), ord=1) / tf.norm(ref_kspace_tensor, ord=1))



loss_check = tf.multiply(scalar, tf.norm(kspaceCheck_tensor - nw_output_kspace) / tf.norm(kspaceCheck_tensor)) + \
       tf.multiply(scalar, tf.norm(kspaceCheck_tensor - nw_output_kspace, ord=1) / tf.norm(kspaceCheck_tensor, ord=1))

all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

##########

global_step = tf.Variable(0, trainable=False)
boundaries = [40, 70]
values = [5e-4, 1e-4, 5e-5]
learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries,
values)

##########




if args.only_cg == True:
    pass
else:
    if args.lr_schedule == False:
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step = global_step)

saver = tf.train.Saver(max_to_keep=100)
sess_trn_filename = os.path.join(directory, 'model')
totalLoss = []
avg_cost = 0
with tf.Session(config=config) as sess:
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    
    sess.run(tf.global_variables_initializer())
    if args.restore == True:
        saver.restore(sess, tf.train.latest_checkpoint(args.saved_model_dir))

    ref_kspace, nw_input, trn_mask, loss_mask, init_cmap, init_imag, cmap_input, _ = gen_mask(sens_maps)
    print('SSDU Parameters: Epochs: ', args.epochs, ', Batch Size:', args.batchSize,
          ', Number of trainable parameters: ', sess.run(all_trainable_vars))
    #feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, sens_mapsP: sens_maps, kspaceCheck: kspace_check}

    print('Training...')
    for ep in range(1, args.epochs + 1):
        if (args.mask_type == 'new_guassian') and ((ep - 1) % args.mask_up_freq == 0):
            ref_kspace, nw_input, trn_mask, loss_mask, init_cmap, init_imag, cmap_input, _ = gen_mask(sens_maps)
        feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, \
                    sens_init : init_cmap, img_init : init_imag, kspaceCheck: kspace_check, \
                    BasisP: basis, Cmap_kspaceP : cmap_input, Cmap_maskP : cmap_mask, pre_cmap: sens_maps, pre_imag: pre_gt}
        sess.run(iterator.initializer, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        try:
            for jj in range(total_batch):
                if args.only_cg == True:
                    Mu1s, Mu2s, Mu3s, Imag0, Cmap0 = sess.run([MU1, MU2, MU3,imag_0, cmap_0])
                else:
                    #tmp, _, _, Images, Cmaps, Kspace_loss, Inters, Mu1s, Mu2s, Mu3s, loss_checks, Imag0, Cmap0, Global_step = sess.run([loss, update_ops, optimizer, nw_output_img, nw_output_cmap, nw_output_kspace, INTER, MU1, MU2, MU3,loss_check, imag_0, cmap_0, global_step])
                    tmp, _, _, Images, Cmaps, Kspace_loss, Inters, Mu1s, Mu2s, Mu3s, loss_checks, Imag0, Cmap0 = sess.run([loss, update_ops, optimizer, nw_output_img, nw_output_cmap, nw_output_kspace, INTER, MU1, MU2, MU3,loss_check, imag_0, cmap_0])
                    avg_cost += tmp / total_batch
                    toc = time.time() - tic
                    totalLoss.append(avg_cost)
                    print("Epoch:", ep, "elapsed_time =""{:f}".format(toc), "cost =", "{:.3f}".format(avg_cost), 'los check', loss_checks)
                print(Mu1s, Mu2s, Mu3s)
                
            if ep == args.epochs:
                print('saving check.mat.....')
                if args.only_cg == True:
                    dicts = {'imag0': Imag0, 'cmap0':Cmap0}
                else:
                    dicts = {'out': Images, 'out_cmap': Cmaps, 'out_kspace':Kspace_loss, 'inters': Inters, 'imag0': Imag0, 'cmap0':Cmap0}
                sio.savemat('Check.mat', dicts)


        except tf.errors.OutOfRangeError:
            pass

        if (np.mod(ep, 1) == 0):
            saver.save(sess, sess_trn_filename, global_step=ep)
            sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})

end_time = time.time()
sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})
print('Training completed in  ', ((end_time - start_time) / 60), ' minutes')
