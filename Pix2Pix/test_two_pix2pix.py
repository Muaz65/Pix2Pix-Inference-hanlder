import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from Util.visualizer import Visualizer
from Util.import html
from PIL import Image
from torchvision import transforms
import tensorflow as tf
import torch
import cv2
#from python.demo import Translationmain
import numpy as np
from matplotlib import pyplot as plt
import os 
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


def pix2pixDriver(model,dataset,visualizer,web_dir,webpage):

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        
        model.set_input(data) 
        model.test()

        visuals = model.get_current_visuals() 
        #print(visuals.shape)   
        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)   
    webpage.save()





def preprocess(image):

    #image = np.concatenate((image, image), axis=1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    A = Image.fromarray(image)
    '''
   
    AB = AB.resize((256 * 2, 256), Image.BICUBIC)
    AB = transforms.ToTensor()(AB)

    w_total = AB.size(2)
    w = int(w_total / 2)    
    h = AB.size(1)     

    w_offset = random.randint(0, max(0, w - 256 - 1))
    h_offset = random.randint(0, max(0, h - 256 - 1))   

    A = AB[:, h_offset:h_offset + 256,
            w_offset:w_offset + 256]
    B = AB[:, h_offset:h_offset + 256,
            w + w_offset:w + w_offset + 256]

    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

    tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
    B = tmp.unsqueeze(0)
    A=A.unsqueeze(0)

    '''

    A = A.resize((256 , 256), Image.BICUBIC)
    A = transforms.ToTensor()(A)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    A=A.unsqueeze(0)





    return {'A': A, 'B':A }


def pix2pixEval(model,image):

    data=preprocess(image)
        
    model.set_input(data) 
    #print(data['A']="lol.jpg")
    model.test()
    #need to replace the viuslas and call the functionn here
    output  = model.get_current_visuals() 
    return output




opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.continue_train = False

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test

# counter=0
# cap = cv2.VideoCapture('football.mp4')
# while(cap.isOpened()):
#   # Capture frame-by-frame
#   ret, frame = cap.read()
#   frame=cv2.resize(frame,(256,256))
#   frame = np.concatenate((frame, frame), axis=1)
#   path='./datasets/soccer_seg_detection/test/frame.jpg'
#   cv2.imwrite(path,frame)
#   driver()

#   Readpath='results/soccer_seg_detection_pix2pix/test_latest/images/frame_fake_D.png'
#   img = cv2.imread(Readpath,0)
#   os.system(cd)
#   os.system("python demo.py")

#   counter+=1
#   print(counter)
#   #cv2.imshow('ModelOutput',frame)


