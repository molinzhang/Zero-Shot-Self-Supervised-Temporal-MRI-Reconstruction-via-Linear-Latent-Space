import tensorflow as tf
import data_consistency as ssdu_dc
import tf_utils
import models.networks as networks
import parser_ops
import scipy.io as sio
import utils
parser = parser_ops.get_parser()
args = parser.parse_args()



class UnrolledNet():
    """

    Parameters
    ----------
    input_x: batch_size x nrow x ncol x 2
    sens_maps: batch_size x ncoil x nrow x ncol

    trn_mask: batch_size x nrow x ncol, used in data consistency units
    loss_mask: batch_size x nrow x ncol, used to define loss in k-space

    args.nb_unroll_blocks: number of unrolled blocks
    args.nb_res_blocks: number of residual blocks in ResNet

    Returns
    ----------

    x: nw output image
    nw_kspace_output: k-space corresponding nw output at loss mask locations

    x0 : dc output without any regularization.
    all_intermediate_results: all intermediate outputs of regularizer and dc units
    mu: learned penalty parameter


    """

    def __init__(self, FhMy, Cmap_kspace, sens_init, imag_init, trn_mask, loss_mask, basis, cmap_mask, pre_cmap, pre_imag, g_method):
        self.FhMy = FhMy # 32, 12, 208, 256, 2
        if args.use_cmapk == True:
            self.Cmap_kspace = Cmap_kspace # 32, 12, 208, 256, 2
        else:
            self.Cmap_kspace = FhMy # 32, 12, 208, 256, 2
        self.sens_init = sens_init[0:1] # 1, 12, 208, 256, 2
        self.imag_init = imag_init # 32, 208, 256, 2
        self.trn_mask = trn_mask # 32, 208, 256
        self.loss_mask = loss_mask # 32, 208, 256
        self.cmap_mask = cmap_mask[0:1] # 0, 12, 208, 256
        self.basis = basis # 32, 4, 1, 1
        self.pre_cmap = pre_cmap[0:1]
        self.pre_imag = pre_imag
        self.g_method = g_method

        self.W, self.H, self.C =  args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB

        if (args.multi_channel == True) or (args.subspace == True):
            self.FhMy = tf.expand_dims(self.FhMy, 0)
            self.Cmap_kspace = tf.expand_dims(self.Cmap_kspace, 0)
            self.imag_init = tf.expand_dims(self.imag_init, 0)
            self.sens_init = tf.expand_dims(self.sens_init, 0)
            self.trn_mask = tf.expand_dims(self.trn_mask, 0)
            self.loss_mask = tf.expand_dims(self.loss_mask, 0)
            self.cmap_mask = tf.expand_dims(self.cmap_mask, 0)
            self.basis = tf.expand_dims(self.basis, 0)
            self.pre_cmap = tf.expand_dims(self.pre_cmap, 0)
            self.pre_imag = tf.expand_dims(self.pre_imag, 0)
            ##
            self.cmap_mask = tf.expand_dims(self.cmap_mask, -1)
        #self.FhMy = tf_utils.tf_complex2real(tf.reduce_sum(tf.conj(self.basis) * \
        #                tf_utils.tf_real2complex(tf.expand_dims(self.input_x, 1)), 0, keepdims = False)) 

        self.model = self.Unrolled_SSDU()

    def denoise_block(self, x, flag):
        if args.multi_channel == True:
            x = tf.transpose(x, perm = [0, 2, 3, 1, 4])
            x = tf.reshape(x, [1, self.W, self.H, -1])
            if flag == 'cmap':
                x = networks.ResNet(x, args.nb_res_blocks_cmap, flag = flag)
            elif flag == 'imag':
                x = networks.ResNet(x, args.nb_res_blocks_imag, flag = flag)
            x = tf.reshape(x, [1, self.W, self.H, -1, 2])
            x = tf.transpose(x, perm = [0, 3, 1, 2, 4])
        elif args.subspace == True:
            x = tf.transpose(x, perm = [0, 2, 3, 1, 4])
            x = tf.reshape(x, [1, self.W, self.H, -1])
            if flag == 'cmap':
                x = networks.ResNet(x, args.nb_res_blocks_cmap, flag = flag)
            elif flag == 'imag':
                x = networks.ResNet(x, args.nb_res_blocks_imag, flag = flag)
            x = tf.reshape(x, [1, self.W, self.H, -1, 2])
            x = tf.transpose(x, perm = [0, 3, 1, 2, 4])
        else:
            if flag == 'cmap':
                x = networks.ResNet(x, args.nb_res_blocks_cmap)
            elif flag == 'imag':
                x = networks.ResNet(x, args.nb_res_blocks_imag)
        return x

    def cg_block(self, rhs, cmap_imag, mu, flag, gradient_method = 'MG'):
        if gradient_method == 'AG':
            x = ssdu_dc.dc_block(rhs, cmap_imag, self.trn_mask, mu, self.basis, flag = flag)
        elif gradient_method == 'MG':
            if flag == 'imag':
                x = ssdu_dc.dcManualGradient_imag(rhs, cmap_imag)
            elif flag == 'cmap':
                x = ssdu_dc.dcManualGradient_cmap(rhs, cmap_imag)
        return x

    def Unrolled_SSDU(self):
        
        all_intermediate_results = [[0 for _ in range(4)] for _ in range(args.nb_unroll_blocks)]
        if args.use_imagrecon == True:
            imag = self.imag_init # 1, 32, 208, 256, 2, basis: 1, 32, 4, 1, 1 
        else:
            imag = self.pre_imag

        if args.subspace == True:
            imag_coeff = tf_utils.tf_real2complex(tf.expand_dims(imag, axis = 2)) # 1, 32, 1, 208, 256
            imag = tf.reduce_sum(imag_coeff * tf.conj(self.basis), axis = 1, keepdims = False)
            imag = tf_utils.tf_complex2real(imag) # 1, coeff, 208, 256, 2

        cmap = self.sens_init # 1, 1, 12, 208, 256, 2

        mu_cmap1 = networks.mu_param_cmap_1()
        mu_cmap2 = networks.mu_param_cmap_2()
        mu_imag1 = networks.mu_param_imag_1()
        if args.use_cmaprecon == True:
            ## Start : compute cmap0 and imag0
            if args.subspace == True:
                rhs_cmap0 = tf_utils.tf_real2complex(tf.expand_dims(imag, axis = 1)) # 1, 1, coeff, 208, 256
            
                rhs_cmap0 = tf.reduce_sum(rhs_cmap0 * self.basis, axis = 2, keepdims = False) # 1, 32, 208, 256. This is the image
                rhs_cmap0 = tf.reduce_sum(tf.expand_dims(tf.conj(rhs_cmap0), axis = 2) * tf_utils.tf_real2complex(self.Cmap_kspace), axis = 1, keepdims = True)
                # 1, 32, 1, 208, 256 * 1, 32, 12, 208, 256 = 1, 1, 12, 208, 256
                rhs_cmap0 = tf_utils.tf_complex2real(rhs_cmap0) # 1, 1, 12, 208, 256, 2
            
            else:
                rhs_cmap0 = tf.reduce_sum(tf.expand_dims(tf.conj(tf_utils.tf_real2complex(imag)), axis = 2) * tf_utils.tf_real2complex(self.Cmap_kspace), axis = 1, keepdims = True)
                rhs_cmap0 = tf_utils.tf_complex2real(rhs_cmap0)


            cmap0 = self.cg_block(rhs_cmap0, imag, (mu_cmap1, mu_cmap2), flag = 'cmap', gradient_method = self.g_method)
            #1, 1, 12, 208, 256, 2

        else:
            cmap0 = self.pre_cmap

        cmap_norm0 = tf_utils.tf_real2complex(cmap0)
        cmap_norm_conj0= tf.conj(cmap_norm0)
        cmap_mag0 = tf.math.sqrt(tf.reduce_sum(cmap_norm0 * cmap_norm_conj0, axis = 2, keepdims = True))
        cmap0 = cmap0 / (tf_utils.tf_complex2real(cmap_mag0)[..., 0:1] + 1e-10) 
        #print(cmap0.shape, self.cmap_mask.shape)
        #(1, 1, 12, 208, 256, 2) (1, ?, ?, ?)
        cmap0 = cmap0 * self.cmap_mask

        if args.use_imagrecon == True:
            
            ### imag0
            rhs_imag0 = tf.reduce_sum(tf.conj(tf_utils.tf_real2complex(cmap0)) * tf_utils.tf_real2complex(self.FhMy), axis = 2, keepdims = False)
            # 1, 32, 208, 256
            if args.subspace == True:
                rhs_imag0 = tf.expand_dims(rhs_imag0, 2)
                rhs_imag0 = tf.reduce_sum(tf.conj(self.basis) * rhs_imag0, axis = 1, keepdims = False)
                # 1, 4, 208 , 256, 2 

            rhs_imag0 = tf_utils.tf_complex2real(rhs_imag0)
            # 1, 32, 208, 256, 2
            imag0 = self.cg_block(rhs_imag0, cmap0, mu_imag1, flag = 'imag', gradient_method = self.g_method)
            # 1, 32, 208, 256, 2
            ## End : compute cmap0 and imag0
        else:
            imag0 = self.pre_imag
            if args.subspace == True:
                imag_coeff0 = tf_utils.tf_real2complex(tf.expand_dims(imag0, axis = 2)) # 1, 32, 1, 208, 256
                imag0 = tf.reduce_sum(imag_coeff0 * tf.conj(self.basis), axis = 1, keepdims = False)
                imag0 = tf_utils.tf_complex2real(imag0) # 1, coeff, 208, 256, 2

                

        if args.only_cg == True:
            ## only cmap0 and imag0 that are important
            return imag0, cmap0, imag0, all_intermediate_results, mu_cmap1, mu_cmap2, mu_imag1, imag0, cmap0


        with tf.name_scope('SSDUModel'):
            with tf.variable_scope('Weights', reuse=tf.AUTO_REUSE):
                for i in range(args.nb_unroll_blocks):
                    if args.order == 'denoise_first':
                        ## recon of cmap
                        if args.use_cmaprecon == True:
                            if args.subspace == True:
                                ## reconstruction of cmap
                                rhs_cmap = tf_utils.tf_real2complex(tf.expand_dims(imag, axis = 1)) # 1, 1, coeff, 208, 256
                                rhs_cmap = tf.reduce_sum(rhs_cmap * self.basis, axis = 2, keepdims = False) # 1, 32, 208, 256. This is the image
                                rhs_cmap = tf.reduce_sum(tf.expand_dims(tf.conj(rhs_cmap), axis = 2) * tf_utils.tf_real2complex(self.Cmap_kspace), axis = 1, keepdims = True)
                                # 1, 32, 1, 208, 256 * 1, 32, 12, 208, 256 = 1, 12, 208, 256
                                rhs_cmap = tf_utils.tf_complex2real(rhs_cmap) # 1, 12, 208, 256, 2
                            else:
                                rhs_cmap = tf.reduce_sum(tf.expand_dims(tf.conj(tf_utils.tf_real2complex(imag)), axis = 2) * tf_utils.tf_real2complex(self.Cmap_kspace), axis = 1, keepdims = True)
                                rhs_cmap = tf_utils.tf_complex2real(rhs_cmap)

                            #rhs_cmap1 = tf.identify(rhs_cmap)
                            #rhs_cmap1 = tf.stop_gradient(rhs_cmap1)

                            cmap_denoise = self.denoise_block(tf.reshape(cmap, [1, self.C, self.W, self.H, 2]), flag = 'cmap')
                            #cmap_denoise = self.denoise_block(tf.reshape(self.pre_cmap, [1, self.C, self.W, self.H, 2]), flag = 'cmap')
                            cmap_denoise = tf.expand_dims(cmap_denoise, 0)
                            mu_cmap1 = networks.mu_param_cmap_1()
                            mu_cmap2 = networks.mu_param_cmap_2()

                            rhs_cmap = tf.stop_gradient(rhs_cmap) + mu_cmap1 * cmap_denoise
                            #rhs_cmap = rhs_cmap + mu_cmap1 * cmap_denoise
                            #rhs_cmap_stop = tf.identity(rhs_cmap)
                            #rhs_cmap_stop = tf.stop_gradient(rhs_cmap_stop)
                            #imag_stop = tf.identity(imag)
                            #imag_stop = tf.stop_gradient(imag_stop)

                            cmap = self.cg_block(rhs_cmap, imag, (mu_cmap1, mu_cmap2), flag = 'cmap', gradient_method = self.g_method)                            
                            ##1, 1, 12, 208, 256, 2

                        else:
                            cmap = self.pre_cmap

                        ## cmap normalization along channel
                        cmap_norm = tf_utils.tf_real2complex(cmap)
                        cmap_norm_conj = tf.conj(cmap_norm)
                        cmap_mag = tf.math.sqrt(tf.reduce_sum(cmap_norm * cmap_norm_conj, axis = 2, keepdims = True))
                        cmap = cmap / (tf_utils.tf_complex2real(cmap_mag)[..., 0:1] + 1e-10) 
                        ##

                        if args.use_imagrecon == True:
                            ## recon of imag
                            rhs_imag = tf.reduce_sum(tf.conj(tf_utils.tf_real2complex(cmap)) * tf_utils.tf_real2complex(self.FhMy), axis = 2, keepdims = False)
                            # 1, 32, 208, 256, 2
                            if args.subspace == True:
                                rhs_imag = tf.expand_dims(rhs_imag, 2)
                                rhs_imag = tf.reduce_sum(tf.conj(self.basis) * rhs_imag, axis = 1, keepdims = False)
                                # 1, coeff, 208, 256, 2
                            rhs_imag = tf_utils.tf_complex2real(rhs_imag)
                            # 1, coeff, 208, 256, 2

                            imag_denoise = self.denoise_block(imag, flag = 'imag')
                            mu_imag1 = networks.mu_param_imag_1()

                            rhs_imag = tf.stop_gradient(rhs_imag) + mu_imag1 * imag_denoise
                            #rhs_imag = rhs_imag + mu_imag1 * imag_denoise
                            #rhs_imag_stop = tf.identity(rhs_imag)
                            #rhs_imag_stop = tf.stop_gradient(rhs_imag_stop)
                            #cmap_stop = tf.identity(cmap)
                            #cmap_stop = tf.stop_gradient(cmap_stop)

                            imag = self.cg_block(rhs_imag, cmap, mu_imag1, flag = 'imag', gradient_method = self.g_method)
                        else:
                            imag = self.pre_imag
                            if args.subspace == True:
                                imag_coeff = tf_utils.tf_real2complex(tf.expand_dims(imag, axis = 2)) # 1, 32, 1, 208, 256
                                imag = tf.reduce_sum(imag_coeff * tf.conj(self.basis), axis = 1, keepdims = False)
                                imag = tf_utils.tf_complex2real(imag) # 1, coeff, 208, 256, 2
                    # ...................................................................................................
                    if args.use_cmaprecon == True:
                        all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(cmap_denoise))
                        all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(cmap))
                    else:
                        all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(cmap))
                        all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(cmap))  
                    if args.use_imagrecon == True:                
                        all_intermediate_results[i][2] = tf_utils.tf_real2complex(tf.squeeze(imag_denoise))
                        all_intermediate_results[i][3] = tf_utils.tf_real2complex(tf.squeeze(imag))
                    else:                
                        all_intermediate_results[i][2] = tf_utils.tf_real2complex(tf.squeeze(imag))
                        all_intermediate_results[i][3] = tf_utils.tf_real2complex(tf.squeeze(imag))
            nw_kspace_output = ssdu_dc.SSDU_kspace_transform(imag, cmap, self.loss_mask, self.basis)
            nw_kspace_output = tf.reshape(nw_kspace_output, [-1, self.C, self.W, self.H, 2]) # [slice, coil channel, 256, 256, 2]
        return imag, cmap, nw_kspace_output, all_intermediate_results, mu_cmap1, mu_cmap2, mu_imag1, imag0, cmap0
