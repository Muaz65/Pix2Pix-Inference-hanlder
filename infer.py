
from Pix2Pix.pix2pix_model_loader import create_pix2pix_model 
from Pix2Pix.opt import Pix2Pix_OPT 
from Pix2Pix.test import *
from utils.utility import *
import glob

pix2pix_model_path = './weights'


# Pix2Pix Module
pix2pixModel = create_pix2pix_model(Pix2Pix_OPT(pix2pix_model_path))



for file in glob.glob("data/*"):
    print(file)



    frame= cv2.imread(file)

    # frame , label= image[:, 0: image.shape[1]//2] , image[:, image.shape[1]//2:]



    pix2pixOutput = pix2pixEval(pix2pixModel, frame)
    camera_test = pix2pixOutput['fake_B']

    camera_test = cv2.cvtColor(camera_test,cv2.COLOR_RGB2BGR)

    concatenated_img = cv2.hconcat([cv2.resize(frame, (1024,1024)), camera_test])

    # cv2.imshow("label", label)

    cv2.imshow("predicted", concatenated_img)
    
    cv2.imwrite( "out/" + file.split("/")[-1] ,concatenated_img)


    cv2.waitKey(1)

