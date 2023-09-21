import tensorflow as tf
import numpy as np
import parser_ops

parser = parser_ops.get_parser()
args = parser.parse_args()

def conv_layer(input_data, conv_filter, is_relu=False, is_scaling=False, flag = 'cmap'):
    """
    Parameters
    ----------
    x : input data
    conv_filter : weights of the filter
    is_relu : applies  ReLU activation function
    is_scaling : Scales the output

    """

    W = tf.get_variable('W'+flag, shape=conv_filter, initializer=tf.random_normal_initializer(0, 0.05))
    x = tf.nn.conv2d(input_data, W, strides=[1, 1, 1, 1], padding='SAME')

    if (is_relu):
        x = tf.nn.relu(x)

    if (is_scaling):
        scalar = tf.constant(0.1, dtype=tf.float32)
        x = tf.multiply(scalar, x)

    return x


def ResNet(input_data, nb_res_blocks, flag):
    """

    Parameters
    ----------
    input_data : nrow x ncol x 2. Regularizer Input
    nb_res_blocks : default is 15.

    conv_filters : dictionary containing size of the convolutional filters applied in the ResNet
    intermediate outputs : dictionary containing intermediate outputs of the ResNet

    Returns
    -------
    nw_output : nrow x ncol x 2 . Regularizer output

    """
    if flag == 'imag':
        conv_filters = dict([('w1'+flag, (3, 3, 2*args.net_cfactor, args.filters_imag)), ('w2'+flag, (3, 3, args.filters_imag, args.filters_imag)), ('w3'+flag, (3, 3, args.filters_imag, 2*args.net_cfactor))])
    elif flag == 'cmap':
        conv_filters = dict([('w1'+flag, (3, 3, 2*args.net_cmap, args.filters_cmap)), ('w2'+flag, (3, 3, args.filters_cmap, args.filters_cmap)), ('w3'+flag, (3, 3, args.filters_cmap, 2*args.net_cmap))])
    intermediate_outputs = {}

    with tf.variable_scope('FirstLayer'+flag):
        intermediate_outputs['layer0'+flag] = conv_layer(input_data, conv_filters['w1'+flag], is_relu=False, is_scaling=False, flag = flag)

    for i in np.arange(1, nb_res_blocks + 1):
        with tf.variable_scope('ResBlock' + str(i)+flag):
            conv_layer1 = conv_layer(intermediate_outputs['layer' + str(i - 1)+flag], conv_filters['w2'+flag], is_relu=True, is_scaling=False, flag = flag)
            conv_layer2 = conv_layer(conv_layer1, conv_filters['w2'+flag], is_relu=False, is_scaling=True, flag = flag)

            intermediate_outputs['layer' + str(i)+flag] = conv_layer2 + intermediate_outputs['layer' + str(i - 1)+flag]

    with tf.variable_scope('LastLayer'+flag):
        rb_output = conv_layer(intermediate_outputs['layer' + str(i)+flag], conv_filters['w2'+flag], is_relu=False, is_scaling=False, flag = flag)

    with tf.variable_scope('Residual'+flag):
        temp_output = rb_output + intermediate_outputs['layer0'+flag]
        nw_output = conv_layer(temp_output, conv_filters['w3'+flag], is_relu=False, is_scaling=False, flag = flag)

    return nw_output


def mu_param_cmap_1():
    """
    Penalty parameter used in DC units, x = (E^h E + \mu I)^-1 (E^h y + \mu * z)
    """

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        mu = tf.get_variable(name='mu_cmap_1', dtype=tf.float32, initializer=args.init_para_cmap1, trainable = True) # 0.05

    return mu

def mu_param_cmap_2():
    """
    Penalty parameter used in DC units, x = (E^h E + \mu I)^-1 (E^h y + \mu * z)
    """

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        mu = tf.get_variable(name='mu_cmap_2', dtype=tf.float32, initializer=args.init_para_cmap2) # 0.05

    return mu

def mu_param_imag_1():
    """
    Penalty parameter used in DC units, x = (E^h E + \mu I)^-1 (E^h y + \mu * z)
    """

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        mu = tf.get_variable(name='mu_imag_1', dtype=tf.float32, initializer=args.init_para_imag, trainable = True) # 0.05

    return mu


def AE_net(input_data, nb_res_blocks):
    """

    Parameters
    ----------
    input_data : nrow x ncol x 2. Regularizer Input
    nb_res_blocks : default is 15.

    conv_filters : dictionary containing size of the convolutional filters applied in the ResNet
    intermediate outputs : dictionary containing intermediate outputs of the ResNet

    Returns
    -------
    nw_output : nrow x ncol x 2 . Regularizer output

    """

    conv_filters = dict([('w1', (3, 3, 2*args.net_cfactor, args.filters)), ('w2', (3, 3, args.filters, args.filters)), ('w3', (3, 3, args.filters, 2*args.net_cfactor))])
    intermediate_outputs = {}

    with tf.variable_scope('FirstLayer'):
        intermediate_outputs['layer0'] = conv_layer(input_data, conv_filters['w1'], is_relu=False, is_scaling=False)

    for i in np.arange(1, nb_res_blocks + 1):
        with tf.variable_scope('ResBlock' + str(i)):
            conv_layer1 = conv_layer(intermediate_outputs['layer' + str(i - 1)], conv_filters['w2'], is_relu=True, is_scaling=False)
            conv_layer2 = conv_layer(conv_layer1, conv_filters['w2'], is_relu=False, is_scaling=True)

            intermediate_outputs['layer' + str(i)] = conv_layer2 + intermediate_outputs['layer' + str(i - 1)]

    with tf.variable_scope('LastLayer'):
        rb_output = conv_layer(intermediate_outputs['layer' + str(i)], conv_filters['w2'], is_relu=False, is_scaling=False)

    with tf.variable_scope('Residual'):
        temp_output = rb_output + intermediate_outputs['layer0']
        nw_output = conv_layer(temp_output, conv_filters['w3'], is_relu=False, is_scaling=False)

    return nw_output