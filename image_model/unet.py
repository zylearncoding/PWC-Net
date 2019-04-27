import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import models

# from keras.models import load_model
##########################################################

# os.chdir("~/shitnet/")
def load_images(folder_name):
    fram_dir = 'Frames/' + folder_name
    seg_dir = 'png/' + folder_name

    x_set = []
    y_set = []
    for filename in os.listdir(fram_dir):
        if filename[0] == '.':
            continue
        x_set.append(filename[0:-3])
    for filename in os.listdir(seg_dir):
        if filename[0] == '.':
            continue
        y_set.append(filename[0:-3])
    x_set.sort()
    y_set.sort()
    x_set_new = []
    for i in x_set:
        if (i in y_set):
            x_set_new.append(i)
    x_set = x_set_new
    img_x = [cv2.imread(fram_dir + i + 'jpg') for i in x_set]
    img_y = [cv2.imread(seg_dir + i + 'png') for i in y_set]
    print("########################")
    print("Images loaded")
    print("########################")
    print("Training set size:  " + str(len(img_x)))
    print("Label set size:    " + str(len(img_y)))
    for i in range(len(img_x)):
        img_x[i] = cv2.resize(img_x[i],(256,256), interpolation = cv2.INTER_CUBIC)/255.0
        img_y[i] = cv2.resize(img_y[i],(256,256), interpolation = cv2.INTER_CUBIC)/255.0

    data = [img_x,img_y]
    return data







print("########################")
print("U-net begin")
print("########################")
def conv_block(x, n_channels, droprate = 0.25):
    """ for UNet """
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_channels, 3,(1,1), padding = 'same', kernel_initializer = 'he_normal')(x) 
    x = Dropout(droprate)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_channels, 3,(1,1), padding = 'same', kernel_initializer = 'he_normal')(x) 
    return x 

def up_block(x, n_channels):
    """ for UNet """
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(size = [2,2])(x)
    x = Conv2D(n_channels, 2,(1,1), padding = 'same', kernel_initializer = 'he_normal')(x)
    return x

def Conv_UNet(x, droprate=0.25):
    """ 1-D Convolutional UNet https://arxiv.org/abs/1505.04597 """

    conv0 = Conv2D(192, 3,(1,1), padding = 'same', kernel_initializer = 'he_normal')(x) 

    conv1 = conv_block(conv0, 128, droprate)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = conv_block(pool1, 192, droprate)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = conv_block(pool2, 384, droprate)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = conv_block(pool3, 512, droprate)

    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
    conv5 = conv_block(pool4, 1024, droprate)
    up5 = conv5

    up4 = up_block(up5, 512)
    up4 = concatenate([conv4,up4], axis = 3)
    up4 = conv_block(up4, 512, droprate)

    up4 = conv4

    up3 = up_block(up4, 384)
    up3 = concatenate([conv3,up3], axis = 3)
    up3 = conv_block(up3, 384, droprate)

    up2 = up_block(up3, 192)
    up2 = concatenate([conv2,up2], axis = 3)
    up2 = conv_block(up2, 192, droprate)

    up1 = up_block(up2, 128)
    up1 = concatenate([conv1,up1], axis = 3)
#     up1 = conv_block(up1, 128, droprate)
    up1 = conv_block(up1, 3, droprate)      ##########

    up1 = BatchNormalization()(up1)
    up1 = ReLU()(up1)

    return up1 


########################################################
batch_size = 3
epochs = 50
img_shape = (256, 256, 3)
        
########################################################
x_train = Input(shape=img_shape)
prediction = Conv_UNet(x_train)

model = models.Model(inputs=x_train, outputs=prediction)
# print("Creat new model")
model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])
# model.summary()

############### model save #################
checkpoint_path = "model_save/unet_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1,period=5)
############################################
if len(os.listdir("model_save/"))==2:
    print("########################")
    print("Creat New Model")
    print("########################")
else:
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    print("########################")
    print("Load Latest Model")
    print("########################")
############################################
print("########################")
print("Training start")
print("########################")

for i in range(1,15):
    if i<10:
        zeros_str = "0000"
    if 9<i<100:
        zeros_str = "000"
    if 99<i<1000:
        zeros_str = "00"
    if 999<i:
        zeros_str = "0"
    folder_name = zeros_str + str(i) + "/"
    print("Load folder" + folder_name)
    img_x,img_y = load_images(folder_name)
    model.fit(x=np.array(img_x[0:-1]),y=np.array(img_y[0:-1]),epochs=15,batch_size=3,callbacks = [cp_callback])
#     loss, accuracy = model.evaluate(x=np.array(img_x[0:-1]),y=np.array(img_y[0:-1]))
#     print(loss)
    


################# save a image ###################
# test_img = model.predict(np.array(img_x[100:103]))
# print(test_img[1])
# cv2.imwrite('tmp/head1.jpg',test_img[1]*255)
