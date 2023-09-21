import tensorflow as tf
import tf_utils
import parser_ops
import models.networks as networks

parser = parser_ops.get_parser()
args = parser.parse_args()


class data_consistency():
    """
    Data consistency class can be used for:
        -performing E^h*E operation in the paper
        -transforming final network output to kspace
    """

    def __init__(self,  mask, basis):
        with tf.name_scope('EncoderParams'):
            self.shape_list = tf.shape(mask)
            self.mask = mask
            self.W, self.H, self.C =  args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB
            if (args.subspace == True) or (args.multi_channel == True):
                self.mask = tf.cast(tf.reshape(self.mask, [-1, 1, self.W, self.H]), dtype = tf.complex64)

            self.basis = basis # 10, 4, 1, 1, 
            self.scalar = tf.complex(tf.sqrt(tf.to_float(self.shape_list[-2] * self.shape_list[-1])), 0.)

    def EhE_imag_Op(self, img, cmap, mu):
        """
        Performs (E^h*E+ mu*I) || Q*x
        """
        with tf.name_scope('EhE_img'):
            #print(cmap.shape)
            if len(cmap.shape) == 3:
                Asxes = 0
                coil_imgs = cmap * img
            elif len(cmap.shape) == 4:
                Asxes = 1
                if args.subspace == True:
                    ## used for coeff and basis
                    imgs = tf.cast(tf.reshape(img, [1, -1, self.W, self.H]), dtype = tf.complex64)
                    img_coeff = tf.reduce_sum(imgs * self.basis, 1, keepdims = True) # 10, 1, 256, 256
                    coil_imgs = cmap * img_coeff
                else:
                    imgs = tf.cast(tf.reshape(img, [-1, 1, self.W, self.H]), dtype = tf.complex64)
                    coil_imgs = cmap * imgs
            
            
            
            kspace = tf_utils.tf_fftshift(tf.fft2d(tf_utils.tf_ifftshift(coil_imgs))) / self.scalar
            masked_kspace = kspace * self.mask

            image_space_coil_imgs = tf_utils.tf_ifftshift(tf.ifft2d(tf_utils.tf_fftshift(masked_kspace))) * self.scalar

            

            if args.subspace == True:
                image_space_comb = tf.reduce_sum(image_space_coil_imgs * tf.conj(cmap), axis=Asxes, keepdims = True)
                image_space_comb = tf.reduce_sum(image_space_comb * tf.conj(self.basis), 0) # 1, coeff, 256, 256    
                        
            else:
                image_space_comb = tf.reduce_sum(image_space_coil_imgs * tf.conj(cmap), axis=Asxes)

            
            ispace = image_space_comb + mu * tf.cast(tf.reshape(imgs, [-1, self.W, self.H]), dtype = tf.complex64)
            
            if len(cmap.shape) == 3:
                ispace = tf.squeeze(ispace)
            
        return ispace

    def EhE_cmap_Op(self, cmap, imag, mu):
        """
        Performs Q * cmap = (E^h*E+ mu_1*I + mu_2*D^h*D) * cmap
        """
        with tf.name_scope('EhE_cmap'):
            mu1, mu2 = mu
            if len(cmap.shape) == 3:
                Asxes = 0
                coil_imgs = cmap * imag
            elif len(cmap.shape) == 4: # 1, 12, W, H
                Asxes = 1
                if args.subspace == True:
                    ## used for coeff and basis
                    imag = tf.cast(tf.reshape(imag, [1, -1, self.W, self.H]), dtype = tf.complex64)
                    imag = tf.reduce_sum(imag * self.basis, 1, keepdims = True) # 32, 1, 256, 256
                    coil_imgs = cmap * imag # 32, 12, 208, 256
                else:
                    imag = tf.cast(tf.reshape(imag, [-1, 1, self.W, self.H]), dtype = tf.complex64)
                    coil_imgs = cmap * imag

            
            kspace = tf_utils.tf_fftshift(tf.fft2d(tf_utils.tf_ifftshift(coil_imgs))) / self.scalar
            masked_kspace = kspace * self.mask

            image_space_coil_imgs = tf_utils.tf_ifftshift(tf.ifft2d(tf_utils.tf_fftshift(masked_kspace))) * self.scalar
            
            ## 32, 12, 208, 256 * 32, 1, 208, 256
            cmap_space_comb = tf.reduce_sum(image_space_coil_imgs * tf.conj(imag), axis= 0, keepdims = True)
            
            
            ### The smoothness constraints
            smooth_cmap = tf_utils.tf_complex2real(cmap)
            smooth_cmap = tf.transpose(smooth_cmap, [0, 1, 4, 2, 3])
            smooth_cmap = tf.reshape(smooth_cmap, [-1, self.W, self.H, 1])
            smooth_cmap = laplacian(smooth_cmap)
            smooth_cmap = tf.reshape(smooth_cmap, [1, self.C, 2, self.W, self.H])
            smooth_cmap = tf.transpose(smooth_cmap, [0, 1, 3, 4, 2])
            smooth_cmap = tf_utils.tf_real2complex(smooth_cmap)

            #print(cmap_space_comb.shape, cmap.shape)
            cspace = cmap_space_comb + mu1 * cmap + mu2 * smooth_cmap
            
            if len(cmap.shape) == 3:
                cspace = tf.squeeze(cspace)
            
        return cspace

    def SSDU_kspace(self, img, cmap):
        """
        Transforms unrolled network output to k-space
        and selects only loss mask locations(\Lambda) for computing loss
        """

        with tf.name_scope('SSDU_kspace'):
            if len(cmap.shape) == 4:
                if args.subspace == True:
                    ## used for coeff and basis
                    img = tf.cast(tf.reshape(img, [1, -1, self.W, self.H]), dtype = tf.complex64)
                    img = tf.reduce_sum(img * self.basis, 1, keepdims = True) # 10, 1, 256, 256
                else:
                    img = tf.cast(tf.reshape(img, [-1, 1, self.W, self.H]), dtype = tf.complex64)
            coil_imgs = cmap * img
            kspace = tf_utils.tf_fftshift(tf.fft2d(tf_utils.tf_ifftshift(coil_imgs))) / self.scalar
            masked_kspace = kspace * self.mask
            
        return masked_kspace

    def Supervised_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        """

        with tf.name_scope('Supervised_kspace'):
            coil_imgs = self.sens_maps * img
            kspace = tf_utils.tf_fftshift(tf.fft2d(tf_utils.tf_ifftshift(coil_imgs))) / self.scalar

        return kspace


def conj_grad(input_elems, mu_param, flag):
    """
    Parameters
    ----------
    input_data : contains tuple of  reg output rhs = E^h*y + mu*z , sens_maps and mask
    rhs = nrow x ncol x 2
    sens_maps : coil sensitivity maps ncoil x nrow x ncol
    mask : nrow x ncol
    mu : penalty parameter

    Encoder : Object instance for performing encoding matrix operations

    Returns
    -------
    data consistency output, nrow x ncol x 2

    """
    rhs, sens_imag, mask, basis = input_elems
    if flag == 'imag':
        mu_param = tf.complex(mu_param, 0.)
    elif flag == 'cmap':
        mu_param1, mu_param2 = mu_param
        mu_param1 = tf.complex(mu_param1, 0.)
        mu_param2 = tf.complex(mu_param2, 0.)
        mu_param = (mu_param1, mu_param2)

    rhs = tf_utils.tf_real2complex(rhs)
    sens_imag = tf_utils.tf_real2complex(sens_imag)

    Encoder = data_consistency(mask, basis)
    if flag == 'imag':
        cond = lambda i, *_: tf.less(i, args.CG_Iter_imag)
    elif flag == 'cmap':
        cond = lambda i, *_: tf.less(i, args.CG_Iter_cmap)

    def body(i, rsold, x, r, p, mu):
        with tf.name_scope('CGIters'):
            if flag == 'imag':
                Ap = Encoder.EhE_imag_Op(p, sens_imag, mu)
            else:
                Ap = Encoder.EhE_cmap_Op(p, sens_imag, mu)
            alpha = tf.complex(rsold / tf.to_float(tf.reduce_sum(tf.conj(p) * Ap)), 0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = tf.to_float(tf.reduce_sum(tf.conj(r) * r))
            beta = rsnew / rsold
            beta = tf.complex(beta, 0.)
            p = r + beta * p

        return i + 1, rsnew, x, r, p, mu
    
    x = tf.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rsold = tf.to_float(tf.reduce_sum(tf.conj(r) * r), )
    loop_vars = i, rsold, x, r, p, mu_param
    cg_out = tf.while_loop(cond, body, loop_vars, name='CGloop_'+flag, parallel_iterations=1)[2]

    return tf_utils.tf_complex2real(cg_out)



def dc_block(rhs, init, mask, mu, basis, flag = 'imag'): # (None, 256, 256, 2) (None, 8, 256, 256) (None, 256, 256)
    """
    DC block employs conjugate gradient for data consistency,
    """

    def cg_map_func(input_elems):
        cg_output = conj_grad(input_elems, mu, flag)

        return cg_output
    
    dc_block_output = tf.map_fn(cg_map_func, (rhs, init, mask, basis), dtype=tf.float32, name='mapCG_'+flag) #map_fu will project the data in dimension

    return dc_block_output


def callCG_cmap(rhs, cmap_imag):
    """
    this function will call the function myCG on each image in a batch
    """
    G=tf.get_default_graph()
    getnext = G.get_operation_by_name('getNext')
    _, _, _, _, trn_mask, _, _, basis, _, cmap_mask, _, _ = getnext.outputs
    if (args.multi_channel == True) or (args.subspace == True):
        trn_mask = tf.expand_dims(trn_mask, 0)
        basis = tf.expand_dims(basis, 0)  
        cmap_mask = tf.expand_dims(cmap_mask, 0)
    #mu =  networks.mu_param_imag_1()
    mu_cmap_1 = networks.mu_param_cmap_1()
    mu_cmap_2 = networks.mu_param_cmap_2()
    mu = (mu_cmap_1, mu_cmap_2)
    #cmap_imag = G.get_tensor_by_name("ExpandDims_2:0")

    def fn(input_elems):
        cg_output = conj_grad(input_elems, mu, 'cmap')
        return cg_output

    if args.use_cmapk == True:
        inp = (rhs, cmap_imag, cmap_mask, basis)
    elif args.use_cmapk == False:
        inp = (rhs, cmap_imag, trn_mask, basis)


    rec=tf.map_fn(fn, inp, dtype=tf.float32, name='mapCG2_cmap')
    return rec

def callCG_imag(rhs, cmap_imag):
    """
    this function will call the function myCG on each image in a batch
    """
    G=tf.get_default_graph()
    getnext = G.get_operation_by_name('getNext')
    _, _, _, _, trn_mask, _, _, basis, _, _, _, _ = getnext.outputs
    if (args.multi_channel == True) or (args.subspace == True):
        trn_mask = tf.expand_dims(trn_mask, 0)
        basis = tf.expand_dims(basis, 0)  

    #print(cmap_imag)

    mu = networks.mu_param_imag_1()
    #cmap_imag = G.get_tensor_by_name("SSDUModel/Weights/ExpandDims_4:0")

    def fn(input_elems):
        cg_output = conj_grad(input_elems, mu, 'imag')
        return cg_output

    inp = (rhs, cmap_imag, trn_mask, basis)
    rec=tf.map_fn(fn, inp, dtype=tf.float32, name='mapCG2_imag')
    return rec

@tf.custom_gradient
def dcManualGradient_cmap(rhs, cmap_imag):
    """
    This function impose data consistency constraint. Rather than relying on
    TensorFlow to calculate the gradient for the conjuagte gradient part.
    We can calculate the gradient manually as well by using this function.
    Please see section III (c) in the paper.
    """
    y=callCG_cmap(rhs, cmap_imag)
    def grad(inp):
        out=callCG_cmap(inp, cmap_imag)
        return (out, tf.zeros_like(cmap_imag))
    
    return y,grad

@tf.custom_gradient
def dcManualGradient_imag(rhs, cmap_imag):
    """
    This function impose data consistency constraint. Rather than relying on
    TensorFlow to calculate the gradient for the conjuagte gradient part.
    We can calculate the gradient manually as well by using this function.
    Please see section III (c) in the paper.
    """
    y=callCG_imag(rhs, cmap_imag)
    def grad(inp):
        out=callCG_imag(inp, cmap_imag)
        return (out, tf.zeros_like(cmap_imag))
    
    return y,grad


def SSDU_kspace_transform(imag, cmap, mask, basis):
    """
    This function transforms unrolled network output to k-space at only unseen locations in training (\Lambda locations)
    """

    imag = tf_utils.tf_real2complex(imag) # 1, 32, 208, 256
    cmap = tf_utils.tf_real2complex(cmap) # 1, 1, 12, 208, 256

    def ssdu_map_fn(input_elems):
        Imag, Cmap, mask_enc, basis = input_elems
        Encoder = data_consistency(mask_enc, basis)
        nw_output_kspace = Encoder.SSDU_kspace(Imag, Cmap)

        return nw_output_kspace

    masked_kspace = tf.map_fn(ssdu_map_fn, (imag, cmap, mask, basis), dtype=tf.complex64, name='ssdumapFn')

    return tf_utils.tf_complex2real(masked_kspace)


def Supervised_kspace_transform(nw_output, sens_maps, mask):
    """
    This function transforms unrolled network output to k-space
    """

    nw_output = tf_utils.tf_real2complex(nw_output)

    def supervised_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc)
        nw_output_kspace = Encoder.Supervised_kspace(nw_output_enc)

        return nw_output_kspace

    kspace = tf.map_fn(supervised_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='supervisedmapFn')

    return tf_utils.tf_complex2real(kspace)

def laplacian(input):
    lap_filter = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 'float32')
    lap_filter = tf.reshape(lap_filter, [3, 3, 1, 1])
    lap_input = tf.nn.conv2d(input, lap_filter,
                          strides=[1, 1, 1, 1], padding='SAME')
    return lap_input

