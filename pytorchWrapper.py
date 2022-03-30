from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image
import os
from skimage import io, transform,color
import time
import datetime

import itertools
from util.image_pool import ImagePool
from models import BaseModel
from models import networks
from models import pix2pix_model

from util.visualizer import Visualizer
from options.test_options_vec import TestOptions as TestOptionsVec
from options.test_options_recon import TestOptions as TestOptionsRec
from models import pix2pix_model1
from util.visualizer import save_images
from util import html,util

import pyVec


def TimestampMillisec64():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)

plt.ion()   # interactive mode

class ToTensor(object):
    def __call__(self, sample):
        data,target,edge = (sample['data'], sample['target'], sample['edge'])
        data=data.transpose((2,0,1))
        target = target.transpose((2,0,1))
        edge=edge.transpose((2,0,1))
        return {'data': torch.from_numpy(data.astype(np.float32)),
                'target': torch.from_numpy(target.astype(np.float32)),
                'edge': torch.from_numpy(edge.astype(np.float32))
                }
class Normalize(object):
    def __call__(self, sample):
        data,target,edge = (sample['data'], sample['target'], sample['edge'])
     #   print(target.shape)
     #   print(target.mean(1).mean(1))
     #   print(target.mean(1).mean(1).shape)
     #   d=data.numpy()
     #   t=target.numpy()
        m=[0.48838824, 0.4546832,  0.4485845 ]
       # s=[0.31811991, 0.30911697, 0.30346411]
        m2=[50.0,0.5,-0.5]
        s=[0.5,0.5,0.5]
        one=[1.0,1.0,1.0]
        s2=[100.0,255.0,255.0]
        s3=[50.0,128.0,128.0]
    #    print (t.std(1).std(1))
        return {'data': transforms.functional.normalize(data,mean=m2,std=s2),
                'target': transforms.functional.normalize(target,mean=m2,std=s2),
                'edge':transforms.functional.normalize(edge,mean=m2,std=s2)
                }
class Normalize2(object):
    def __call__(self, sample):
        data,target,edge = (sample['data'], sample['target'], sample['edge'])
     #   print(target.shape)
     #   print(target.mean(1).mean(1))
     #   print(target.mean(1).mean(1).shape)
     #   d=data.numpy()
     #   t=target.numpy()
        m=[0.48838824, 0.4546832,  0.4485845 ]
       # s=[0.31811991, 0.30911697, 0.30346411]
        m2=[50.0,0.5,-0.5]
        s=[0.5,0.5,0.5]
        one=[1.0,1.0,1.0]
        s2=[100.0,255.0,255.0]
        s3=[50.0,128.0,128.0]
    #    print (t.std(1).std(1))
        return {'data': transforms.functional.normalize(data,mean=m2,std=s3),
                'target': transforms.functional.normalize(target,mean=m2,std=s3),
                'edge':transforms.functional.normalize(edge,mean=m2,std=s3)
                }


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, sample):
        data, target, edge = (sample['data'], sample['target'], sample['edge'])
        h, w = (512,512)
        new_h, new_w = (self.output_size, self.output_size)
   #     print(data.shape)
   #     print((h,w,new_h,new_w))
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)

        data = data[top:top+new_h,
                    left:left+new_w]
        target = target[top:top+new_h,
                    left:left+new_w]
        edge = edge[top:top+new_h,
                    left:left+new_w]
        return {'data': data, 'target':target, 'edge':edge}

class Pad(object):
    def __call__(self, sample):
        data, target, edge = (sample['data'], sample['target'], sample['edge'])
        h, w = (512,512)

     #   data = cv.resize(data, dsize=(1024, 1024), interpolation=cv.INTER_CUBIC)
     #   target = cv.resize(target, dsize=(1024, 1024), interpolation=cv.INTER_CUBIC)
     #   edge = cv.resize(edge, dsize=(1024, 1024), interpolation=cv.INTER_CUBIC)

        print(data.shape)
        data=np.pad(data,((20,),(20,),(0,)),'edge')
        print(data.shape)
        target=np.pad(target,((20,),(20,),(0,)),'edge')
        edge=np.pad(edge,((20,),(20,),(0,)),'edge')

        return {'data': data, 'target':target, 'edge':edge}

class RandomRotate(object):
    def __call__(self, sample,degree):
        data, target = (sample['data'], sample['target'])
        h, w = (512,512)
        
   #     print(data.shape)
   #     print((h,w,new_h,new_w))
   #     degree = np.random.randint(0,360)
            

        data = transform.rotate(data,degree)
        target = transform.rotate(target,degree)
        return {'data': data, 'target':target}

class RandomFlip(object):
    def __call__(self, sample):
        data, target = (sample['data'], sample['target'])
   #     print(data.shape)
   #     print((h,w,new_h,new_w))
        shouldFlip = np.random.randint(2)
        if shouldFlip>0:
            data = np.flip(data,axis=1)
            target = np.flip(target,axis=1)
        return {'data': data, 'target':target}

class Flip(object):
    def __call__(self, sample):
        data, target = (sample['data'], sample['target'])
   #     print(data.shape)
   #     print((h,w,new_h,new_w))
        data = np.flip(data,axis=1)
        target = np.flip(target,axis=1)
        return {'data': data, 'target':target}

class MyRescale(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, sample):
        data, target = (sample['data'], sample['target'])
        new_h, new_w = (self.output_size, self.output_size)
      #  print(data.shape)
      #  print(data)
        data_s = transform.resize(data, (new_h, new_w))
      #  print(data_s)
        target_s =transform.resize(target, (new_h, new_w))
        return {'data': data_s, 'target':target_s}



class DanborooDataSetVectorized(Dataset):
    def __init__(self,root_dir,transform=None):
       # if data:
       #     self.dataOrTarget='data/'
       # else:
       #     self.dataOrTarget='target/'
        self.filenames = [f for f in os.walk(root_dir+'data/')][0][2]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,idx):
       # img_name_data = self.root_dir+'0150/'+self.filenames[idx].rstrip("_fake_B.png")+".jpg"
       # img_name_target = self.root_dir+'0150/'+self.filenames[idx].rstrip("_fake_B.png")+".jpg"

    #    img_name_data = self.root_dir+'colorSources/'+self.filenames[idx].rstrip(".png")+"_colorSource.png"
    #    img_name_target = self.root_dir+'target/'+self.filenames[idx].rstrip(".png")+".png"
    #    img_name_edges = self.root_dir+'edgesFakeB/'+self.filenames[idx].rstrip(".png")+"_edges.png"
        img_name_data = self.root_dir+'colorSourcesRealB/'+self.filenames[idx].rstrip(".png")+"_colorSource.png"
        img_name_target = self.root_dir+'target/'+self.filenames[idx].rstrip(".png")+".png"
        img_name_edges = self.root_dir+'edgesRealB/'+self.filenames[idx].rstrip(".png")+"_edges.png"
      #  img_name_edges = self.root_dir+'edgesRealB/'+self.filenames[idx].rstrip(".png")+"_edges.png"
       # img_name_target = self.root_dir+'target/'+self.filenames[idx].rstrip(".png")+".png"
       # img_name_data = self.root_dir+'dataFakeB/'+self.filenames[idx].rstrip(".png")+"_fake_B.png"
        data = io.imread(img_name_data)
        if data.shape!=(512,512,3):
            data = np.zeros((512,512,3))
        #print(data.shape)
        data =data.astype(np.float32)/255.0
       # data = np.clip(data,0,1)  
        data = color.rgb2lab(data)
       # print(data)
      #  print(color.lab2rgb(data))

        target = io.imread(img_name_target)
        if target.shape!=(512,512,3):
            target = np.zeros((512,512,3))
        target = target.astype(np.float32)/255.0
       # target = np.clip(target,0,1)  
        target = color.rgb2lab(target)

        edge = io.imread(img_name_edges)
        edge = (edge.astype(np.float32)/255.0)
       # edge = np.clip(edge,0,1)  
        edge = color.rgb2lab(edge)
       # data=data.transpose((2,0,1))
       # target = target.transpose((2,0,1))
        sample = {'data':data.astype(np.float64),'target':target.astype(np.float64),'edge':edge.astype(np.float64)}

        flips= Flip()(sample)
        rotations = [Pad()(sample)]
       # rotations = [sample for i in range(1)]
      #  rotations = [RandomCrop(256)(sample) for i in range(10)]
       # rotations +=[RandomRotate()(rotations,i*20) for i in range(18)]
        tensors = [Normalize()(ToTensor()(rot)) for rot in rotations]
       # tensors = [ToTensor()(rot) for rot in rotations]
       # print(len(tensors))
        sample= {'data': torch.stack([d['data'] for d in tensors]), 
        'target': torch.stack([d['target'] for d in tensors]),'A_Path':img_name_data.rstrip('.png'),
        'B_Path':img_name_target.rstrip('.png'),'edge':torch.stack([d['edge'] for d in tensors]),'A2_Path':img_name_edges.rstrip('.png')}
      #  print(sample['data'].size())

        
        #if self.transform:
        #    sample= self.transform(sample)
        return sample

def getEdgeMap(inputImg):
    time_a=TimestampMillisec64()
    img_np=np.asarray(
        bytearray(inputImg.file.read())
        , dtype=np.uint8)
    img=cv.imdecode(img_np,cv.IMREAD_COLOR)

    print(img.shape)
    data =img.astype(np.float32)/255.0  
    data = color.rgb2lab(data)
    sample = {'data':data.astype(np.float64),'target':data.astype(np.float64),'edge':data.astype(np.float64)}
    flips= Flip()(sample)
    rotations = [sample]
    tensors = [Normalize()(ToTensor()(rot)) for rot in rotations]
    sample= {'data': torch.stack([d['data'] for d in tensors])}


    datas= sample['data']
    datas.to(device)
    time_c=TimestampMillisec64()
    modelVec.set_input((datas,datas,'',''))
    modelVec.test()           # run inference
    time_d=TimestampMillisec64()
    print(time_d-time_c)
    visuals = modelVec.get_current_visuals()  # get image results
    img_path = 'WebUI/edgeMap'     # get image paths

    res=visuals['fake_B']
    im = util.tensor2im(res)
    #####doing the LAB 2 RGB transform########
    save_path = os.path.join(img_path, inputImg.filename)
    util.save_image(im, save_path)
    
    res=pyVec.trackContour(im,img,"WebUI/json/"+inputImg.filename.rstrip('.jpg').rstrip('.png')+".json")
    getReconstruction(res[0],res[1],inputImg.filename)
    util.save_image(res[0], 'WebUI/bEdgeMap/'+inputImg.filename)
    util.save_image(res[1], 'WebUI/colorSourceMap/'+inputImg.filename)
    time_b=TimestampMillisec64()
    print(time_b-time_a)
    return

def getReconstruction(edgeMap,colorSourceMap,filename):
    time_a=TimestampMillisec64()

    data = colorSourceMap.astype(np.float32)/255.0
    data = color.rgb2lab(data)

    edge = edgeMap.astype(np.float32)/255.0  
    edge = color.rgb2lab(edge)


    sample = {'data':data.astype(np.float64),'target':data.astype(np.float64),'edge':edge.astype(np.float64)}
    flips= Flip()(sample)
    rotations = [Pad()(sample)]
    tensors = [Normalize2()(ToTensor()(rot)) for rot in rotations]
    sample= {'data': torch.stack([d['data'] for d in tensors]),
    'edge':torch.stack([d['edge'] for d in tensors])}


    datas= sample['data']
    print(datas.shape)
    #datas.to(device)
    edges= sample['edge']
    #edges.to(device)

    time_c=TimestampMillisec64()
    modelRecon.set_input((datas,datas,edges,'','',''))
    modelRecon.test()           # run inference
    time_d=TimestampMillisec64()
    print(time_d-time_c)
    visuals = modelRecon.get_current_visuals()  # get image results
    img_path = 'WebUI/reconstruction'     # get image paths

    res=visuals['fake_B']
    im = util.tensor2im2(res)
    #####doing the LAB 2 RGB transform########
    save_path = os.path.join(img_path, filename)
    util.save_image(im, save_path)
    edgeMap=None
    csMap=None
    time_b=TimestampMillisec64()
    print(time_b-time_a)
   
    return 

csMap=None
edgeMap=None
def setCsMap(input):
    global csMap,edgeMap
    img_np=np.asarray(
        bytearray(input.file.read())
        , dtype=np.uint8)
    img=cv.imdecode(img_np,cv.IMREAD_COLOR)
    img=cv.cvtColor(img,cv.COLOR_RGB2BGR)
    csMap=img
    res=None
    #print(csMap.shape)
    if not (edgeMap is None):
        getReconstruction(edgeMap,csMap,input.filename)
        return True
    return False
def setEdgeMap(input):
    global edgeMap,csMap
    img_np=np.asarray(
        bytearray(input.file.read())
        , dtype=np.uint8)
    img=cv.imdecode(img_np,cv.IMREAD_COLOR)
    img=cv.cvtColor(img,cv.COLOR_RGB2BGR)
    edgeMap=img
    #print(edgeMap.shape)
    res=None
    if not (csMap is None):
        getReconstruction(edgeMap,csMap,input.filename)
        return True
    return False

print("pix2pix")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
optV= TestOptionsVec().parse()
optV.num_threads = 0   # test code only supports num_threads = 1
optV.batch_size = 1    # test code only supports batch_size = 1
optV.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
optV.no_flip = True    # no flip; comment this line if results on flipped images are needed.
optV.display_id = -1
optV.isTrain =False
optV.model='pix2pix'

modelVec = pix2pix_model.Pix2PixModel(optV)
modelVec.setup(optV)
modelVec.eval()

optR= TestOptionsRec().parse()
optR.num_threads = 0   # test code only supports num_threads = 1
optR.batch_size = 1    # test code only supports batch_size = 1
optR.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
optR.no_flip = True    # no flip; comment this line if results on flipped images are needed.
optR.display_id = -1
optR.isTrain =False
optR.model='pix2pix'


modelRecon = pix2pix_model1.Pix2PixModel(optR)
modelRecon.setup(optR)
modelRecon.eval()

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    print("test")
    dataset = DanborooDataSetVectorized('preprocessed/')  # create a dataset given opt.dataset_mode and other options
    data_loader = DataLoader(dataset,batch_size=1,shuffle=True, num_workers=4)
    model = pix2pix_model2.Pix2PixModel(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
 #   model.netG.model.model[0].register_forward_hook(get_activation('ext_conv1'))

    

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    print("test")
    if opt.eval:
        model.eval()
    for i, data in enumerate(data_loader):
     #   if i >= opt.num_test:  # only apply our model to opt.num_test images.
     #       break
     #   activation={}
        datas= data['data']
       # print(datas.size())
        bs, ncrops, c, h, w = datas.size()
        datas = datas.view(-1, c, h, w).to(device)
        targets = data['target']
        targets = targets.view(-1, c, h, w).to(device)
        edges = data['edge']
        edges = edges.view(-1, c, h, w).to(device)

        model.set_input((datas,targets,edges,data['A_Path'],data['B_Path'],data['A2_Path']))
 #       model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
 #       act = activation['ext_conv1'].squeeze()
 #       print(act)
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML