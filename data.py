from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os

class make_train_data(object):
    def __init__(self, dir_path, tempset=100, Tmin=1.0, Tmax=5.0):
        images, labels = [], []
        T = np.linspace(Tmin,Tmax,tempset)

        for i in range(tempset):
            directory = dir_path + '/Temp{:.2f}_{}/'.format(T[i],i)
            files = os.listdir(directory)
            label = np.array([0]*tempset)
            label[i] = 1
            for file in files:
                im = Image.open(directory+file)
                pixels = np.array(im.convert('L').getdata())
                images.append(pixels/255.0)
                labels.append(label)
          
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
        
        class train:
            def __init__(self):
                self.images = []
                self.labels = []
                    
        class test:
            def __init__(self):
                self.images = []
                self.labels = []
                
        self.train = train()
        self.test = test()
                
        self.train.images = np.array(train_images)
        self.train.labels = np.array(train_labels)
        self.test.images = np.array(test_images)
        self.test.labels = np.array(test_labels)