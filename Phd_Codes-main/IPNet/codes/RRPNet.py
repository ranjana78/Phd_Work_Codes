
#---------------------------------Importing Packages----------------------------
import scipy
import datetime
import sys
from scipy import stats
import statistics
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
#import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from iteration_utilities import flatten
from geopy.distance import geodesic
import csv
import os
import time
import cv2
import sys
from sklearn import metrics
import shutil
import random
import math
import PIL
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from PIL import Image
from numpy import asarray
from numpy import array
from numpy import linalg as LA
from collections import Counter
from numpy import dot
from numpy.linalg import norm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from functools import reduce
import json
from itertools import combinations
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy import misc,ndimage
import multiprocessing as mp
import pywt
import pywt.data
import statistics
from statistics import mean
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
os.environ['CUDA_VISIBLE_DEVICES']='0'

#------------------------Count parameters in model------------------------------
def count_parameters(model):
  return sum(p.numel() for p in model.parameters())
#-------------------------------------------------------------------------------
#-----------------------------Read Img Function---------------------------------
#reading of images
def read_img(A):

    datax = []
    datay = []
    C = os.listdir(A)
    for character in C:
        images = os.listdir(A + character + '/')
        c=0
        for img in images:
          image = cv2.resize(cv2.imread(A + character + '/' + img),(84,84))
          datax.append(image)
          datay.append(character)
          c=c+1
          print(c)

          

    return np.array(datax), np.array(datay)
#-------------------------------------------------------------------------------
#--------------------------------Read Directory---------------------------------
def read_images(base_directory):
    """
    Reads all the alphabets from the base_directory
    Uses multithreading to decrease the reading time drastically
    """
    datax = None
    datay = None
    #pool = mp.Pool(mp.cpu_count())
    r,r1 =read_img(base_directory)
    return r,r1
#-------------------------------------------------------------------------------
#------------------Extraction of Query and Support Samples----------------------
def extract_sample(n_way, n_support, n_query, datax, datay):
  """
  Picks random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
  Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
  """
  sample = []
  K = np.random.choice(np.unique(datay), n_way, replace=False)
  for cls in K:
    datax_cls = datax[datay == cls]
    perm = np.random.permutation(datax_cls)
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
  sample = np.array(sample)
  sample = torch.from_numpy(sample).float()
  sample1 = sample.permute(0,1,4,2,3)
  return({
      'original':sample,
      'images': sample1,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query
      })
#-------------------------------------------------------------------------------
#---------------------------------Model Definition------------------------------
class Proto_Conv4(nn.Module):
  def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
  def __init__(self, in_channels, out_channels,H_Dim,H_Dim1,H_Dim2):
      super(Proto_Conv4, self).__init__()

      self.conv1 = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )

      self.conv2 = nn.Sequential(
          nn.Conv2d(H_Dim, out_channels, kernel_size=2, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )

      self.conv3 = nn.Sequential(
          nn.Conv2d(H_Dim1, out_channels, kernel_size=2, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )

      self.conv4 = nn.Sequential(
          nn.Conv2d(H_Dim2, out_channels, kernel_size=2, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )

  def forward(self, x):
      #-------------------------------First Layer-------------------------------
      x1 = self.conv1(x)
      #-------------------------------Second Layer------------------------------
      x2 = self.conv2(x1)
      #-------------------------------Third Layer-------------------------------
      x3 = self.conv3(x2)
      #-------------------------------Fourth Layer------------------------------
      x4 = self.conv4(x3)
      #-------------------------------Flatten Layer-----------------------------
      flattened_x4 = x4.flatten(start_dim=1)

      return flattened_x4
  def set_forward_loss(self, sample,M):
    """
    Computes loss, accuracy and output for classification task
    Args:
        sample (torch.Tensor): shape (n_way, n_support+n_query, (dim))
    Returns:
        torch.Tensor: shape(2), loss, accuracy and y_hat
    """
    sample_images = sample['images'].cuda()
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]
    #---------------------target indices are 0 ... n_way-1----------------------
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda()
    
    #----------encode images of the support and the query set-------------------
    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                   x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)
    z = M(x).to('cuda')
    #------------------------model visualization--------------------------------
    '''dot = make_dot(z, params=dict(model.named_parameters()))
    dot.render("model", format="png")
    #print("Visualization saved as model.png")'''
    #---------------------------------------------------------------------------
    z_dim = z.size(-1)
    z_samples = z[:n_way*n_support]
    z_samples=z_samples.reshape(n_way,n_support,z_dim)
    z_samples = Variable(z_samples, requires_grad=True)
    z_query = z[n_way*n_support:]
    z_query = Variable(z_query, requires_grad=True)
    #----------------------------Making of prototypes---------------------------
    z_proto = CalcluatePrototype_RRP( z_samples )
    #-----------------calculationg distance from each prototypes----------------
    dists = euclidean_dist(z_query,z_proto)

    #--------------compute probabilities,loss and accuracy----------------------
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    return z_samples,z_query,loss_val,{
        'loss': loss_val.item(),
        'acc': acc_val.item(),
        'y_hat': y_hat
        }

#-----------------Function to Calculate Prototpe--------------------------------
def CalcluatePrototype_RRP( z_samples ):
  z_proto=[]
  for i in z_samples:
    P_Vec=RRP(i)
    z_proto.append(P_Vec)
  z_proto=torch.stack(z_proto)
  return z_proto
#---------------------RRPNET Function---------------------
def RRP(Sam):
  Set_A=[]
  Set_B=[]
  for i in Sam:
    Set_A.append(i)
    Set_B.append(i)
  W_Vec=[]
  W_Alpha=[]
  for i,j in zip(Set_A,range(len(Set_A))):
    Set_B.pop(j)
    S_Set_B=torch.stack(Set_B)
    Int_proto=torch.mean(S_Set_B,dim=0)
    Int_dis=euclidean_distance(i,Int_proto)
    Set_B=Set_A.copy()
    Alpha=torch.div(1,Int_dis)
    Weight_Vec=torch.mul(Alpha,i)
    W_Vec.append(Weight_Vec)
    W_Alpha.append(Alpha)
  W_Vec=torch.stack(W_Vec)
  W_Alpha=torch.stack(W_Alpha)
  Weight_Proto=torch.div(torch.mean(W_Vec, dim=0),torch.mean(W_Alpha))
  return Weight_Proto
def euclidean_distance(a, b): 
    # Ensure both tensors are of the same shape assert 
    a.size() == b.size(), "Tensors must have the same shape" 
    # Compute element-wise squared differences 
    diff_sq = (a - b)**2 
    # Sum of squared differences 
    sum_diff_sq = torch.sum(diff_sq) 
    # Euclidean distance 
    euclidean_dist = torch.sqrt(sum_diff_sq) 
    return euclidean_dist 
#-----------------------calculation of euclidean distance-----------------------
def euclidean_dist(x, y):
  """
  Computes euclidean distance btw x and y
  Args:
      x (torch.Tensor): shape (n, d). n usually n_way*n_query
      y (torch.Tensor): shape (m, d). m usually n_way
  Returns:
      torch.Tensor: shape(n, m). For each query, the distances to each centroid
  """
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)
#---------------------------------Training--------------------------------------
def train(M, optimizer, train_x, train_y,n_way, n_support, n_query, max_epoch, epoch_size,PATH_Model):
  """
  Trains the protonet
  Args:
      model
      optimizer
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
  """
  #divide the learning rate by 2 at each epoch, as suggested in paper
  scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
  epoch = 0 #epochs done so far
  stop = False #status to know when to stop
  EE=datetime.datetime.now()
  count=0
  while epoch < max_epoch and not stop:
    running_loss = 0.0
    running_acc = 0.0
    for episode in range(epoch_size):
      sample = extract_sample(n_way, n_support, n_query,train_x, train_y)
      optimizer.zero_grad()
      z_samples,z_query,loss_val,output=M.set_forward_loss(sample,M)
      running_loss += output['loss']
      running_acc += output['acc']
      loss_val.backward()
      optimizer.step()
      #print('Epoch {:d} episode {:d}'.format(epoch,episode))
    epoch_loss = running_loss / epoch_size
    epoch_acc = running_acc / epoch_size

    print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc))

    epoch=epoch+1
    scheduler.step()
  EE1=datetime.datetime.now()
  print('***********Training Time******************:',EE1-EE)
  torch.save(M.state_dict(), PATH_Model)
  best_state = M.state_dict()
  return best_state
#---------------------------------Testing---------------------------------------
def test(M,test_x, test_y, n_way, n_support, n_query, test_episode):
  """
  Tests the protonet
  Args:
      model: trained model
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
  """

  avg_auc=0
  running_loss = 0.0
  running_acc = 0.0
  Test_Acc_rec=[]
  for episode in range(test_episode):
    sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
    z_samples,z_query,loss_val,output=M.set_forward_loss(sample,M)

    running_loss += output['loss']
    running_acc += output['acc']
    Test_Acc_rec.append(output['acc'])
  print('Testing ACC',running_acc/test_episode)
  print('Testing Loss',running_loss/test_episode)
  return running_acc/test_episode,running_loss/test_episode,Test_Acc_rec
#-------------------------------------------------------------------------------
#------------------------------------Pickling of Images-------------------------

'''PATH='/home/ranjana/New_pickle_files/Pickle_84_84/'
Data='200X'
trainx, trainy= read_images('/home/ranjana/Datasets/Current_datasets/Breast-cancer-current/Breast-cancer-resolution/Breast-cancer-'+Data+'/Train_aug/')
testx, testy = read_images('/home/ranjana/Datasets/Current_datasets/Breast-cancer-current/Breast-cancer-resolution/Breast-cancer-'+Data+'/Test/')
with open(PATH+'BC-'+Data+"_trainx.pkl","wb") as f:
  pickle.dump(trainx,f)
with open(PATH+'BC-'+Data+"_trainy.pkl","wb") as f:
  pickle.dump(trainy,f)
with open(PATH+'BC-'+Data+"_testx.pkl","wb") as f:
  pickle.dump(testx,f)
with open(PATH+'BC-'+Data+"_testy.pkl","wb") as f:
  pickle.dump(testy,f)'''
#-------------------------------------------------------------------------------

#--------------------------------Main-------------------------------------------


def main():
  #--------------------------------Seed values------------------------------------
  Np_Seed1=int(sys.argv[1])
  Torch_Seed1=int(sys.argv[2])
  Res=str(sys.argv[3])
  n_way =int(sys.argv[4])
  n_support =int(sys.argv[5])
  n_support1 =int(sys.argv[6])
  max_epoch =int(sys.argv[7])
  Data='BC-'+Res
  np.random.seed(Np_Seed1)
  torch.manual_seed(Torch_Seed1)
  #--------------------------Parameters inside model------------------------------
  Dim_channel=3
  Actual_Hid_Dim=64
  Hid_Dim=64
  Hid_Dim1=64
  Hid_Dim2=64
  #---------------------------parameters to change--------------------------------
  n_query = 5
  epoch_size = 2000
  test_episode = 400
  PATH='/home/ranjana/Refined-Pickle-Files/Pickle-84_84/'
  PATH_Model='/home/ranjana/New_Models/Journal/RRPNET/'+'RRPNET_CONV4_'+Data+'_n_way_'+str(n_way)+'_n_support1_'+str(n_support1)+'.pth'

  #-------------------------------------------------------------------------------

  start=datetime.datetime.now()
  model = Proto_Conv4(in_channels=Dim_channel, out_channels=Actual_Hid_Dim,H_Dim=Hid_Dim,H_Dim1=Hid_Dim1,H_Dim2=Hid_Dim2).to('cuda')

  optimizer = optim.Adam(model.parameters(), lr = 0.1)

  with open(PATH+Data+"_trainx.pkl","rb") as f:
      trainx = pickle.load(f)
  with open(PATH+Data+"_trainy.pkl","rb") as f:
      trainy = pickle.load(f)
  with open(PATH+Data+"_testx.pkl","rb") as f:
      testx = pickle.load(f)
  with open(PATH+Data+"_testy.pkl","rb") as f:
      testy = pickle.load(f)

  print("*************************Training**********************************************")
  best_state=train(model, optimizer,trainx, trainy,n_way, n_support, n_query, max_epoch, epoch_size,PATH_Model)

  print("*************************Testing**********************************************")
  model.load_state_dict(best_state)
  model.eval()

  T_Acc,T_Loss,Test_Acc_rec=test(model,testx, testy,n_way, n_support1, n_query,test_episode)
  end=datetime.datetime.now()
  print("^^^^^^^^^^^^^^Average Accuracy:^^^^^^^^^^^^^^^^^^^^", T_Acc)
  
  print("total time taken:",end-start)
  num_params = count_parameters(model)
  print(f"Number of parameters in the model: {num_params}")
if __name__ == "__main__":
    main()
