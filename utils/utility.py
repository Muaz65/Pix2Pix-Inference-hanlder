from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2

import pyflann
flann = pyflann.FLANN()

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def updateRadar( radar, ball, color):


#     cv2.circle(radar, (int((ball[0]/74)*370), int((ball[1]/115)*575)), 8, color, -1)

#     return radar

def updateRadar( startLoc , radar, color):

    startX,startY=startLoc
    

    cv2.circle(radar, (int((startY/74)*370), int((startX/115)*575)), 8, color, -1)

    return radar

def returnOnlyHomographtMatrix(database, feat, query_image, writeCheck, counter):
    template_h = 74  # yard, soccer template
    template_w = 115
    # t1=time.time()
    model_points = database['points']
    model_line_index = database['line_segment_index']
    databaseFeatures = database['features']
    # database_cameras = database['cameras']
    homographyFeatures=database['HomographyMatrix']

    # Step 2: retrieve a camera using deep features
    
    result, _ = flann.nn(
        databaseFeatures, feat, 1, algorithm="kdtree", trees=1, checks=265)
    retrieved_index = result[0]
    #print("feature debug##############################################")
   # print(databaseFeatures[retrieved_index])
    #print("feaure Debug###############################################")

    """
    Retrieval camera: get the nearest-neighbor camera from database
    """
    retrieved_h = homographyFeatures[retrieved_index]

    # print(time.time()- t1, "in  Flaans")

    return retrieved_h


def directHomographyTranslation(homographyMatrix, centers):
    '''
    Centers format:
    np.array([np.array([[centers[0], centers[1], 1], [centers[0], centers[1], 1]]])])
    # '''
    centers = np.array([[centers[0], centers[1], 1]])
    locs = np.matmul(np.linalg.inv(homographyMatrix), centers.T)
    locs = locs/locs[2, :]
    locs = locs[:2, 0][::-1]
    return (locs[0], locs[1])

  

def CenterNetBallProcessor(detector, frame, ballLoc, deep_sort, globalBallLocation, TeamAids=[], TeamBids=[]):
    ballfoundCheck = False
    # output = returnCenterNetOutput(detector, frame)
    # model, device = load_model()
    output = detect(frame, detector[0], detector[1])


    balls = output[1]
    balls = np.array(balls)

    # players = output[2]
    # refs = output[1]

    #players = [*players, *refs]

    scores = []
    dets = []
    center1=(0,0)
    maxIndex = 0


    if(len(balls) > 0):
        if (len(balls > 0)):

            for outputs in balls:

                bbox = outputs[:4]
                conf = outputs[4]
                bbox = np.array(bbox, dtype=np.int32)
                scores.append(conf)
                dets.append(bbox)

            if(len(scores) > 1):
                maxIndex = scores.index(max(scores))
            else:
                maxIndex = 0

            # maximum confidence ball being used
            if(scores[maxIndex] > 0.5):
                ballfoundCheck = True
                
                # ballPlane = np.zeros_like((115,74,3))       

                bbox = dets[maxIndex]
                # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
                # cv2.imshow("frame", frame)
                # cv2.waitKey(1)

                ballCrop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                globalBallLocation = ((bbox[2]-bbox[0]/2) + bbox[0], (bbox[3] - bbox[1] /2) + bbox[1])

                height, width = ballCrop.shape[:-1]
                center1 = (int(width/2)+bbox[0], int(height/2)+bbox[1])

                center = calculateCoordinates(frame, center1)
                ballLoc=center

    # print("wwwwwww")
    return ballLoc, frame, ballfoundCheck, center1
