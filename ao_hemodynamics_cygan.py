
import numpy as np
from tqdm import tqdm
from colorama import Fore
import logging, os
import math
import scipy.io as io
import scipy 
import random
import argparse

logging.disable(logging.WARNING)

import tensorflow as tf



parser = argparse.ArgumentParser()
parser.add_argument("--flag")
#parser.add_argument("--path2")
args = parser.parse_args()
flag = args.flag

def encoder_layer(x_con, iterations, name,training, pool=True):
   
    with tf.name_scope("encoder_block_{}".format(name)):

        for i in range(iterations):
            x = tf.keras.layers.Conv3D(32,kernel_size=[3,3,3],padding='SAME')(x_con)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x_con = tf.concat([x,x_con], axis = -1)

        if pool is False:
            return x_con
       
        
        pool = tf.keras.layers.AveragePooling3D(pool_size = [2,2,1], strides=[2,2,1],data_format='channels_last')(x_con)

        return x_con, pool



def decoder_layer(input_, x, ch, name, upscale = [2,2,2]):
       
    up = tf.keras.layers.Conv3DTranspose(filters=32,kernel_size = [2,2,1],strides = [2,2,1],padding='SAME',name='upsample'+str(name), use_bias=False)(input_)
    up = tf.concat([up,x], axis=-1, name='merge'+str(name))
    return up

def decoder_layer2(input_, x, ch, name, upscale = [2,2,2]):
       
    up = tf.keras.layers.Conv3DTranspose(filters=32,kernel_size = [2,2,1],strides = [2,2,1],padding='SAME',name='upsample'+str(name), use_bias=False)(input_)
    up = tf.concat([up,x], axis=-1, name='merge'+str(name))
    return up

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv3D(filters, kernel_size=[3,3,3], padding='same'
                             ))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

    return result

def Disc():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[None, None,None, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None,None, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(8, 2, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(16, 2)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(32, 2)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv3D(64, 4, strides=1,padding='SAME'
                                #kernel_initializer=initializer,
                                )(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv3D(1, 4, strides=1,padding='SAME'
                                )(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def DiscBA():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[None, None,None, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None,None, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(8, 2, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(16, 2)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(32, 2)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv3D(64, 4, strides=1,padding='SAME'
                                #kernel_initializer=initializer,
                                )(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv3D(1, 4, strides=1,padding='SAME'
                                )(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def dense_model():

    input_ = tf.keras.layers.Input(shape=(None,None,None,1))
    
    
    conv1,pool1 = encoder_layer(input_,iterations=2,name="encode_im"+str(1),training=True, pool=True)

    conv2, pool2 = encoder_layer(pool1,iterations=4,name="encode_im"+str(2),training=True, pool=True)
    conv3, pool3 = encoder_layer(pool2,iterations=6,name="encode_im"+str(3),training=True, pool=True)
    conv4 = encoder_layer(pool3,iterations=8,name="encode_im"+str(4),training=True, pool=False)
    
    up1 = decoder_layer(conv4,conv3,10,name=12)
    conv7 = encoder_layer(up1,iterations=6,name="conv_im"+str(6),training=True, pool=False)
    up2 = decoder_layer(conv7,conv2,8,name=21)
    conv8 = encoder_layer(up2,iterations=4,name="encode_im"+str(7),training=True, pool=False)
    up3 = decoder_layer(conv8,conv1,6,name=32)
    conv9 = encoder_layer(up3,iterations=2,name="encode_im"+str(8),training=True, pool=False)
    
    
    conv10 = tf.keras.layers.Conv3D(3,(1,1,1),name='logits_re_im',padding='SAME')(conv9)
    

    return tf.keras.Model(inputs=input_,outputs=conv10)

def dense_modelBA():

    input_ = tf.keras.layers.Input(shape=(None,None,None,3))
    
    
    conv1,pool1 = encoder_layer(input_,iterations=2,name="encode"+str(1),training=True, pool=True)

    conv2, pool2 = encoder_layer(pool1,iterations=4,name="encode"+str(2),training=True, pool=True)
    conv3, pool3 = encoder_layer(pool2,iterations=6,name="encode"+str(3),training=True, pool=True)
    conv4 = encoder_layer(pool3,iterations=8,name="encode"+str(4),training=True, pool=False)
    up1 = decoder_layer(conv4,conv3,10,name=41)
    conv7 = encoder_layer(up1,iterations=6,name="conv"+str(6),training=True, pool=False)
    up2 = decoder_layer(conv7,conv2,8,name=211)
    conv8 = encoder_layer(up2,iterations=4,name="encode"+str(7),training=True, pool=False)
    up3 = decoder_layer(conv8,conv1,6,name=321)
    conv9 = encoder_layer(up3,iterations=2,name="encode"+str(8),training=True, pool=False)
    
    
    conv10 = tf.keras.layers.Conv3D(1,(1,1,1),name='logits_re',padding='SAME')(conv9)
    
    return tf.keras.Model(inputs=input_,outputs=conv10)



def parser(tfrecord):

    feature_s = tf.io.parse_single_example(tfrecord,{'test/image': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
               #'test/mag': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
               'test/label': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
               'test/depth': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
               'test/height': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
               'test/width': tf.io.FixedLenFeature(shape=[], dtype=tf.int64)})
    


    height = tf.cast(feature_s["test/height"], tf.int32)
    
    width = tf.cast(feature_s["test/width"], tf.int32) #zeros-padded dim
    depth = tf.cast(feature_s["test/depth"], tf.int32) #zeros-padded dim
    
    # Convert the image data from string back to the numbers
    image = tf.io.decode_raw(feature_s['test/image'], tf.float32) #real component of data
    
    label = tf.io.decode_raw(feature_s['test/label'], tf.float32) #ground-truth



    # Reshape image data into the original shape
    image = tf.reshape(image, [height, width, depth])



    label = tf.reshape(label, [height, width, depth,3])
    label = tf.cast(label,tf.float32)

    image = tf.cast(image,tf.float32)
   
    
    
    return image, label,depth, height, width

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def train_step(Disc,net2,DiscBA,netBA, input_, groundtruth, optimizer, height, width, depth,tt,optimizer1,optimizer2,optimizer3,flag):
    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape,tf.GradientTape() as disc_tapeBA,tf.GradientTape() as gen_tapeBA:
        
        input_ = np.squeeze(input_)
        input_2 = input_.copy()
        
        groundtruth = np.squeeze(groundtruth)
        c2 = input_
        d2 = groundtruth
        
        c2= tf.compat.v1.image.resize_image_with_crop_or_pad(c2, 128, 96) #cropping for input
        d2 = tf.transpose(d2,perm=[3, 0, 1,2])
        d2 = tf.compat.v1.image.resize_image_with_crop_or_pad(d2, 128, 96)
        d2 = tf.transpose(d2,perm=[1, 2, 3,0])
      
        real_c = tf.expand_dims(c2,axis=0)
        real_c = tf.expand_dims(real_c,axis=-1)

        d2 = tf.expand_dims(d2,axis=0)
        #d2 = tf.expand_dims(d2,axis=-1)
       
        
        cc1 = net2(real_c) #2nd CNN
        cc1_ce = netBA(cc1)

        dd2_ce = netBA(d2)
        dd2_ce_m = net2(dd2_ce)

       
        
        
        d2 = tf.squeeze(d2) #Ground-truth
        
        c2 = tf.expand_dims(c2,axis=-1)
        
        mag = cc1
        mse = tf.keras.losses.MeanSquaredError() #Take mean-square error
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        d2 = tf.expand_dims(d2,axis=0)
        disc_real_output = Disc([real_c, d2], training=True)
        disc_generated_output = Disc([real_c, cc1], training=True)

        disc_real_output_m = DiscBA([d2, real_c], training=True)
        disc_generated_output_m = DiscBA([d2, dd2_ce], training=True)

        d2 = tf.squeeze(d2)
        dd2_ce_m = tf.squeeze(dd2_ce_m)
        
        real_c = tf.squeeze(real_c)
        cc1_ce = tf.squeeze(cc1_ce)

        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        real_loss_m = loss_object(tf.ones_like(disc_real_output_m), disc_real_output_m)

        generated_loss_m = loss_object(tf.zeros_like(disc_generated_output_m), disc_generated_output_m)

        total_disc_loss_m = real_loss_m + generated_loss_m

        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        gan_loss_m = loss_object(tf.ones_like(disc_generated_output_m), disc_generated_output_m)

        loss1 = tf.reduce_mean(tf.abs(real_c - cc1_ce))*2
        

        loss1_m = tf.reduce_mean(tf.abs(d2 - dd2_ce_m))*2



        total_cycle_loss = loss1 + loss1_m
        if flag == 'test':
            io.savemat('./ao_hemodynamics/ml1_sim_'+str(tt)+'.mat',{'data':mag.numpy()})
            io.savemat('./ao_hemodynamics/gt_sim_'+str(tt)+'.mat',{'tr':d2.numpy()})
        

        mae = tf.keras.losses.MeanAbsoluteError()
        print(mag.shape)
        print(d2.shape)
        print(real_c.shape)
        print(dd2_ce.shape)
        dd2_ce = tf.squeeze(dd2_ce)

        loss_fn =  generated_loss+total_cycle_loss+ 100*mse(mag,d2) # Modified for additional loss function 
        loss_fn_m = mae(real_c,dd2_ce)+ generated_loss_m+total_cycle_loss #+y_xl+x_zl+y_zl
        

        del mag
        del d2
        
    if flag == 'training': #Backpropgagtion
        variables_d = Disc.trainable_variables #+ net.trainable_variables #All parameters being trained in CNN(s)    
        variables = net2.trainable_variables #+ net.trainable_variables #All parameters being trained in CNN(s)
        variables_BA = netBA.trainable_variables
        variables_dBA = DiscBA.trainable_variables
       
        gradients = gen_tape.gradient(loss_fn, variables) #Establish training gradient
        gradients_d = disc_tape.gradient(total_disc_loss, variables_d) #Establish training gradient
        gradients_BA = gen_tapeBA.gradient(loss_fn_m, variables_BA) #Establish training gradient
        gradients_dBA = disc_tapeBA.gradient(total_disc_loss_m, variables_dBA) #Establish training gradient

        optimizer.apply_gradients(zip(gradients,variables)) #Optimizer
        optimizer1.apply_gradients(zip(gradients_d,variables_d)) #Optimizer
        optimizer2.apply_gradients(zip(gradients_BA,variables_BA)) #Optimizer
        optimizer3.apply_gradients(zip(gradients_dBA,variables_dBA)) #Optimizer
    return loss_fn 


#Example of input list of tfrecords
dataset = tf.data.TFRecordDataset([


'Path/to/training/data/training_data.tfrecords']) #Adjust location of training data


dataset = dataset.map(map_func=parser, num_parallel_calls=3) #parser function for reading tfrecords, estabished above in code
dataset = dataset.batch(1) #batch-size
dataset = dataset.prefetch(40) #pre-fetch data to be queue
dataset = dataset.repeat(500) #Number of epochs to be repeated
dataset = dataset.apply(tf.data.experimental.ignore_errors()) #ignore errors when tf cant read data/skip that data
iterator = iter(dataset)

model = Disc()
model_im = dense_model()
model_1 = DiscBA()
model_im1 = dense_modelBA()
model_im.summary()

import tensorflow.keras as keras

#tt = list(next_element)
class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate



llr = LinearDecay(0.0002, 100*1049, 50*1049)
op = tf.keras.optimizers.Adam(learning_rate=llr) #Using Adam optimizer and static learning rate
op1 = tf.keras.optimizers.Adam(learning_rate=llr) #Using Adam optimizer and static learning rate

discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer_dBA = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

ckpt_im = tf.train.Checkpoint(generator_g=model_im,
                           generator_f=model_im1,
                           discriminator_x=model,
                           discriminator_y=model_1,
                           generator_g_optimizer=op,
                           generator_f_optimizer=op1,
                           discriminator_x_optimizer=discriminator_optimizer,
                           discriminator_y_optimizer=discriminator_optimizer_dBA)




manager_im = tf.train.CheckpointManager(ckpt_im, './ao_hemodynamics_weight', max_to_keep=20)

ckpt_im.restore(manager_im.latest_checkpoint)


if manager_im.latest_checkpoint:
    print("Restored from {}".format(manager_im.latest_checkpoint))
else:
    print("Initializing from scratch.")

tt = 0

for epoch in range(1):

    for check_point in tqdm(range(994),#407    
    bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):

        
        real,d,dep, height, width  = iterator.get_next()
        
        ccc = real
        
        
        
        d = tf.expand_dims(d,axis=4)
        tt = tt+1
        loss = train_step(model,model_im,model_1,model_im1,ccc, d, op, height, width, dep,tt,discriminator_optimizer,op1,discriminator_optimizer_dBA,flag)
        
        if int(check_point) % 497 == 0:
            
            print("loss {:1.2f}".format(loss.numpy()))
            
            
    
    if flag == 'training':
        save_path = manager.save()
        save_path_im = manager_im.save()

        
