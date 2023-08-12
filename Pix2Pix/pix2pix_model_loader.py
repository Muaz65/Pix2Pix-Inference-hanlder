# from .two_pix2pix_model import TwoPix2PixModel
from .pix2pix_model import Pix2PixModel
import time

def create_pix2pix_model(opt):
    model = None
    # model = TwoPix2PixModel()
    model=Pix2PixModel()
    model.initialize(opt)
    # model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    return model
