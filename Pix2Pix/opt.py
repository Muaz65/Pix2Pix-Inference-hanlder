import cv2
import queue
# from Model_Paths.model_paths import pix2pix_model_path


class Pix2Pix_OPT:
    def __init__(self, pix2pix_model_path):
        self.aspect_ratio =  1.0
        self.batchSize =  16
        self.checkpoints_dir =  pix2pix_model_path
        self.continue_train = False
        self.dataroot =  './datasets/soccer_seg_detection'
        self.dataset_mode =  'aligned'
        self.display_id =  1
        self.display_port =  8097
        self.display_winsize =  256
        self.fineSize =  1024
        self.gpu_ids =  [0]
        self.how_many =  186
        self.init_type =  'normal'
        self.input_nc =  3
        self.isTrain =  False
        self.loadSize =  1024
        self.max_dataset_size =  'inf'
        self.model =  'two_pix2pix'
        self.nThreads =  1
        self.n_layers_D =  3
        self.name =  ''
        self.ndf =  64
        self.ngf =  64
        self.no_dropout =  False
        self.no_flip =  True
        self.norm =  'batch'
        self.ntest =  'inf'
        self.output_nc =  3
        self.phase =  'test'
        self.resize_or_crop =  'resize_and_crop'
        self.results_dir =  './results/'
        self.serial_batches =  True
        self.which_direction =  'AtoB'
        self.which_epoch =  'latest'
        self.which_model_netD =  'basic'
        self.which_model_netG =  'unet_256'


