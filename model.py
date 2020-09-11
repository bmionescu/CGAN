
import tensorflow as tf
from layer_functions import *

#________________________

def discriminator(discrim_inputs,discrim_targets,reuse):
    with tf.variable_scope("discriminator", reuse=reuse):
        inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)      
        layer1 = lrelu(discriminator_convolutional_layer(1,inputs, 64, 2), 0.2)
        layer2 = lrelu(batchnorm(discriminator_convolutional_layer(2,layer1, 128, 2)),0.2)
        layer3 = lrelu(batchnorm(discriminator_convolutional_layer(3,layer2, 256, 2)),0.2)
        layer4 = lrelu(batchnorm(discriminator_convolutional_layer(4,layer3, 512, 1)),0.2)
        layer5 = tf.math.sigmoid(discriminator_convolutional_layer(5,layer4, 1, 1))
            
        return layer5

# # # 
    
def generator(generator_inputs,batch_size):
    with tf.variable_scope("generator"):
        # encoder 
        layer1 = generator_convolutional_layer(1,generator_inputs,64)
        layer2 = batchnorm(generator_convolutional_layer(2,lrelu(layer1,0.2),128))
        layer3 = batchnorm(generator_convolutional_layer(3,lrelu(layer2,0.2),256))   
        layer4 = batchnorm(generator_convolutional_layer(4,lrelu(layer3,0.2),512))
        layer5 = batchnorm(generator_convolutional_layer(5,lrelu(layer4,0.2),512))     
        layer6 = batchnorm(generator_convolutional_layer(6,lrelu(layer5,0.2),512))
        layer7 = batchnorm(generator_convolutional_layer(7,lrelu(layer6,0.2),512))     
        layer8 = batchnorm(generator_convolutional_layer(8,lrelu(layer7,0.2),512))
        
        # decoder
        layer9 = tf.nn.dropout(batchnorm(generator_transposed_conv_layer(9,layer8,[batch_size,2,2,512])),keep_prob=0.5)
        layer9 = tf.concat([layer9, layer7], axis=3)
        
        layer10 = tf.nn.dropout(batchnorm(generator_transposed_conv_layer(10,tf.nn.relu(layer9),[batch_size,4,4,512])),keep_prob=0.5)
        layer10 = tf.concat([layer10, layer6], axis=3)
        
        layer11 = tf.nn.dropout(batchnorm(generator_transposed_conv_layer(11,tf.nn.relu(layer10),[batch_size,8,8,512])),keep_prob=0.5)
        layer11 = tf.concat([layer11, layer5], axis=3)
        
        layer12 = batchnorm(generator_transposed_conv_layer(12,tf.nn.relu(layer11),[batch_size,16,16,512]))
        layer12 = tf.concat([layer12, layer4], axis=3)
        
        layer13 = batchnorm(generator_transposed_conv_layer(13,tf.nn.relu(layer12),[batch_size,32,32,256]))
        layer13 = tf.concat([layer13, layer3], axis=3)
        
        layer14 = batchnorm(generator_transposed_conv_layer(14,tf.nn.relu(layer13),[batch_size,64,64,128]))
        layer14 = tf.concat([layer14, layer2], axis=3)
        
        layer15 = batchnorm(generator_transposed_conv_layer(15,tf.nn.relu(layer14),[batch_size,128,128,64]))
        layer15 = tf.concat([layer15, layer1], axis=3)
        
        layer16 = tf.tanh(generator_transposed_conv_layer(16,tf.nn.relu(layer15),[batch_size,256,256,3])) # 3 = RGB
        
        return layer16
        
#________________________

