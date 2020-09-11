
import tensorflow as tf

#________________________

def generator_transposed_conv_layer(layer_number,batch_input,out_shape):

    with tf.variable_scope('gen_transposed_conv_'+str(layer_number)):	     
	    init = tf.random_normal_initializer(0, 0.02)	
	    W_conv = tf.get_variable("W_tconv_"+str(layer_number),shape=[4,4,out_shape[-1],batch_input.shape[-1]],initializer=init)

	    return tf.nn.conv2d_transpose(batch_input,W_conv,output_shape=out_shape,strides=[1,2,2,1],padding='SAME')
    
# # #  
    
def generator_convolutional_layer(layer_number,batch_input,out_channels):
    
    with tf.variable_scope('gen_conv_' + str(layer_number)):	     
	    init = tf.random_normal_initializer(0, 0.02)	
	    W_conv = tf.get_variable("W_conv_" + str(layer_number),shape=[4,4,batch_input.shape[-1],out_channels],initializer=init)

	    return tf.nn.conv2d(batch_input,W_conv,strides=[1,2,2,1],padding='SAME')

# # #
        
def discriminator_convolutional_layer(layer_number,batch_input, out_channels, stride):    
    
    with tf.variable_scope('disc_conv_' + str(layer_number)):	  
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        init = tf.random_normal_initializer(0, 0.02)	
        W_conv = tf.get_variable("W_conv_" + str(layer_number),shape=[4,4,batch_input.shape[-1],out_channels],initializer=init)

        return tf.nn.conv2d(padded_input,W_conv,strides=[1,stride,stride,1],padding='VALID')    
    
# # # 
        
def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
    
# # #
    
def lrelu(x,a):
    with tf.name_scope("lrelu"):
        return tf.maximum(x, a*x)
    
#________________________
    
