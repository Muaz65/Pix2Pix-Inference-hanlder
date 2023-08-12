import os
from PIL import Image
from torchvision import transforms
import cv2
import time
import torchvision.transforms as transforms


def preprocess(image):
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
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    A = Image.fromarray(image)    
    A = A.resize((1024 , 1024), Image.BICUBIC)
    A = transforms.ToTensor()(A)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    A=A.unsqueeze(0)
    return {'A': A, 'B':A }


def pix2pixEval(model,image):


#     t1=time.time()

    data=preprocess(image)
    model.set_input(data) 
    model.test()
    #need to replace the viuslas and call the functionn here
    output  = model.get_current_visuals() 

#     print(time.time()- t1, "in pc2px")

    return output
