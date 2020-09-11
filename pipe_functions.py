
import cv2
import glob
import numpy as np
import re
import random

#________________________

# Sorts filenames: [1.png,10.png,11.png,...,2.png,20.png,21.png,22.png,...] --> [1.png,2.png,3.png,...]
def globsorter(array):
    sorted_array,interim=[],[]
    for i in range(0,len(array)):
        temp = int(re.findall('(\d{1,10})(?=\.)', array[i])[0])
        interim += [[temp,array[i]]]
    interim.sort(key=lambda x: x[0])
    sorted_array=(np.asarray(interim))[:,1].tolist()

    return sorted_array

# # #

# Loads, normalizes and flattens sorted images, puts them in a numpy array  
def load(path,image_size):        
    globb=globsorter(glob.glob(path))
    image_data = []
    for i in range(0,len(globb)):
        img = cv2.resize(cv2.imread(globb[i]),(image_size,image_size)).astype(np.float32)/255         
        image_data += [img]
        
    image_data = np.array(image_data, dtype=np.float32)

    return image_data

# # # 

# Prepares a batch of images from the test data and labels
def test_batch(test_input_data,test_target_data,batch_size,ep,i):
    r = random.sample(range(len(test_input_data)),batch_size)

    test_data_batch = np.asarray([test_input_data[x] for x in r])
    test_target_batch = np.asarray([test_target_data[x] for x in r])

    return test_data_batch, test_target_batch, r	

#________________________
    




