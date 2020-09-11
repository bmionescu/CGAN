
import tensorflow as tf
from model import discriminator,generator
from pipe_functions import globsorter,load,test_batch

#________________________

# Inputs and (hyper)parameters

epochs,savestep = 50,20
batch_size,image_size = 1,256
learning_rate, beta1 = 0.0002,0.5
gan_weight,l1_weight = 1.0,100.0
EPS = 1e-12

inputs_path,targets_path = "train_data/*","train_labels/*"
input_data,target_data = load(inputs_path,256), load(targets_path,256)

test_inputs_path,test_targets_path = "test_data/*","test_labels/*"
test_input_data,test_target_data = load(test_inputs_path,256), load(test_targets_path,256)

print("Loading complete.")

inputs = tf.placeholder("float",[batch_size,image_size,image_size,3])
targets = tf.placeholder("float",[batch_size,image_size,image_size,3])

# # # 

# Loss function

generator_outputs=generator(inputs,batch_size)

disc_prediction_real = discriminator(inputs,targets,False)
disc_prediction_fake = discriminator(inputs,generator_outputs,True)

generator_loss_GAN = tf.reduce_mean(-tf.math.log(disc_prediction_fake + EPS))
generator_loss_L1 = tf.reduce_mean(tf.abs(targets - generator_outputs))

generator_loss = generator_loss_GAN * gan_weight + generator_loss_L1 * l1_weight
discrim_loss = tf.reduce_mean(-(tf.math.log(disc_prediction_real + EPS) + tf.math.log(1 - disc_prediction_fake + EPS)))

# # #

# Optimization

trainable_vars = tf.trainable_variables()

discriminator_vars = [var for var in trainable_vars if 'discriminator' in var.name]
generator_vars = [var for var in trainable_vars if 'generator' in var.name]

disc_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(discrim_loss, var_list=discriminator_vars)
gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(generator_loss, var_list=generator_vars)

# # #

# Main

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()

for ep in range(epochs):
	for i in range(int(len(input_data)/batch_size)):
		sess.run(disc_optimizer,feed_dict={inputs:input_data[batch_size*i:batch_size*(i+1)],
						targets:target_data[batch_size*i:batch_size*(i+1)]})
		sess.run(gen_optimizer,feed_dict={inputs:input_data[batch_size*i:batch_size*(i+1)],
						targets:target_data[batch_size*i:batch_size*(i+1)]})

		if (i+ep*int(len(input_data)/batch_size))%savestep==0:
			disc_loss_evaluated = sess.run(discrim_loss,feed_dict={inputs:input_data[batch_size*i:batch_size*(i+1)],
										targets:target_data[batch_size*i:batch_size*(i+1)]})
			gen_loss_evaluated = sess.run(generator_loss,feed_dict={inputs:input_data[batch_size*i:batch_size*(i+1)],
										targets:target_data[batch_size*i:batch_size*(i+1)]})

			print("epoch: " + str(ep) + ", iteration: " + str(i) + ", losses: " + 
					str(disc_loss_evaluated) + ", " + str(gen_loss_evaluated))
			saver.save(sess,"./saved_model/savemodel.ckpt", global_step=1000)

			test_data_batch,test_target_batch, r = test_batch(test_input_data,test_target_data,batch_size,ep,i)
			test_result = sess.run(generator_outputs,feed_dict={inputs:test_data_batch, targets:test_target_batch})

   			stacked=np.hstack((test_result[0]*255,test_target_data[r[0]]*255))
    			cv2.imwrite("./images/epoch_" + str(ep) + "_iter_" + str(i) + ".png", stacked)


