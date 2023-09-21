import numpy as np
import tensorflow as tf
import tensorflow.io as tfio
c = np.random.randn(1, 5, 5, 1)
a = tf.constant(c, tf.float32)

def laplacian(image, size):
    
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    fil = np.ones([size, size])
    fil[int(size/2), int(size/2)] = 1.0 - size**2
    fil = tf.convert_to_tensor(fil, tf.float32)
    fil = tf.stack([fil]*1, axis=2)
    fil = tf.expand_dims(fil, 3)
    
    print(fil)
    
    result = tf.nn.depthwise_conv2d(image, fil, strides=[1, 1, 1, 1], padding="SAME")
    return result
    
lap = laplacian(a, 3)

b = tf.Session().run(lap)
print(b[0,:,:,0])
print(c[0,:,:,0])
