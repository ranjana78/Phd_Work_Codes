
#---------------------------------Importing Packages----------------------------
import scipy
import datetime
import sys
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
from scipy import stats
import random
import math
import PIL
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
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
    datax1 = []
    datay2 = []
    C = os.listdir(A)
    for character in C:
        print(character)
        images = os.listdir(A + character + '/')
        c=0
        for img in images:
          if ".DS" in img:
            os.remove(A + character + '/' + img)
          else:
            image = cv2.resize(cv2.imread(A + character + '/' + img),(84,84))
            image1 = cv2.resize(cv2.imread(A + character + '/' + img,cv2.IMREAD_GRAYSCALE),(84,84))

            datax.append(image)
            datay.append(character)
            datax1.append(image1)
            datay2.append(character)
            c=c+1
            print(c)



    return np.array(datax), np.array(datay),np.array(datax1), np.array(datay2)
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
    r,r1,r2,r3 =read_img(base_directory)
    return r,r1,r2,r3
#-------------------------------------------------------------------------------
#------------------Extraction of Query and Support Samples----------------------
def extract_sample(n_way, n_support, n_query, datax, datay, datax1, datay1):
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
  samples=[]
  K = np.random.choice(np.unique(datay), n_way, replace=False)
  for cls in K:
    datax_cls = datax[datay == cls]
    datax_cls1 = datax1[datay1 == cls]

    perm_idx = np.random.permutation(len(datax_cls))

    # Apply the same permutation to both datasets
    sample_cls = datax_cls[perm_idx[:(n_support+n_query)]]
    sample_cls1 = datax_cls1[perm_idx[:(n_support+n_query)]]

    sample.append(sample_cls)
    samples.append(sample_cls1)
  sample = np.array(sample)
  sample1 = torch.from_numpy(sample).float()

  sample2 = sample1.permute(0,1,4,2,3)

  samples = np.array(samples)
  samples1 = torch.from_numpy(samples).float()

  return({
      'original1':samples,
      'images1': samples1,
      'original':sample,
      'images': sample2,
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
      self.Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

      self.Avgpool1 = nn.AvgPool2d(kernel_size=3, stride=4,padding=1)

      self.Avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2,padding=1)

  def forward(self, x):
      #-------------------------------First Layer-------------------------------
      x1 = self.conv1(x)
      Tensor_I=[]
      Mean_tensor=[]
      for i in x1:
        I=torch.mean(i,dim=0)
        I = I.unsqueeze(0)
        concatenated_I = torch.cat((i,I),dim=0)
        Tensor_I.append(concatenated_I)
        Mean_tensor.append(I)
      Tensor_I=torch.stack(Tensor_I, dim=0)
      Mean_tensor=torch.stack(Mean_tensor, dim=0)
      #-------------------------------------------------------------------------
      #-------------------------------Second Layer------------------------------
      x2 = self.conv2(Tensor_I)
      Tensor_I=[]
      Mean_tensor1=[]
      for i,j in zip(x2,Mean_tensor):
        I=torch.mean(i,dim=0)
        I = I.unsqueeze(0)
        j=self.Avgpool(j)
        concatenated_I = torch.cat((j,i,I),dim=0)
        Tensor_I.append(concatenated_I)
        Mean_tensor1.append(I)
      Tensor_I=torch.stack(Tensor_I, dim=0)
      Mean_tensor1=torch.stack(Mean_tensor1, dim=0)
      #-------------------------------------------------------------------------
      #-------------------------------Third Layer-------------------------------
      x3 = self.conv3(Tensor_I)
      Tensor_I=[]
      for i,j,k in zip(x3,Mean_tensor,Mean_tensor1):
        j=self.Avgpool1(j)
        k=self.Avgpool2(k)
        I=torch.mean(i,dim=0)
        I = I.unsqueeze(0)
        concatenated_I = torch.cat((I,i,k,j),dim=0)
        Tensor_I.append(concatenated_I)
      Tensor_I=torch.stack(Tensor_I, dim=0)
      #-------------------------------------------------------------------------
      #-------------------------------Fourth Layer------------------------------
      x4 = self.conv4(Tensor_I)
      #-------------------------------------------------------------------------
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
    sample_images1 = sample['images1'].cuda()
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]
    #---------------------target indices are 0 ... n_way-1----------------------
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda()
    #---------------------------Wavelet purpose---------------------------------
    sample_orig = sample['original']
    sample_orig1 = sample['original1']
    x_support = sample_orig[:, :n_support]
    x_query = sample_orig[:, n_support:]
    x_support1 = sample_orig1[:, :n_support]
    x_query1 = sample_orig1[:, n_support:]
    Supp1,Query1=Wavelet_conv(x_support,x_query,x_support1,x_query1)
    Supp1 = torch.from_numpy(Supp1).float()
    Supp1 = Supp1.permute(0,1,4,2,3)
    Supp1 = Variable(Supp1, requires_grad=True)
    Supp1 = Supp1.cuda()
    Query1 = torch.from_numpy(Query1).float()
    Query1 = Query1.permute(0,1,4,2,3)
    Query1 = Variable(Query1, requires_grad=True)
    Query1 = Query1.cuda()
    #---------------------------------------------------------------------------
    #----------encode images of the support and the query set-------------------
    x = torch.cat([Supp1.contiguous().view(n_way * n_support, *Supp1.size()[2:]),
                   Query1.contiguous().view(n_way * n_query, *Query1.size()[2:])], 0)
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
    z_proto = CalcluatePrototype_Mean( z_samples )
    #-----------------calculationg distance from each prototypes----------------
    dists = euclidean_dist(z_query,z_proto)

    #--------------compute probabilities,loss and accuracy----------------------
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
    # Assuming log_p_y, target_inds, n_way, and n_query are defined as in your snippet
    # Convert log probabilities to probabilities
    probs = torch.exp(log_p_y)
    # Flatten the tensors to get a 1D array for each
    probs_flat = probs.view(-1, n_way)
    true_labels_flat = target_inds.view(-1)

    # Convert tensors to numpy arrays for use with sklearn metrics
    probs_np = probs_flat.cpu().detach().numpy()
    true_labels_np = true_labels_flat.cpu().detach().numpy()
    if n_way==2:
      P=probs_np[:, 1]

      Auc = roc_auc_score(true_labels_np, P)
    else:
      Auc = roc_auc_score(true_labels_np,probs_np,average="macro", multi_class='ovo')

    # Get predicted class labels
    _, preds = probs_flat.max(1)
    preds_np = preds.cpu().detach().numpy()

    # Calculate Precision, Recall, and F1-Score
    precision = precision_score(true_labels_np, preds_np, average='macro')
    recall = recall_score(true_labels_np, preds_np, average='macro')
    f1 = f1_score(true_labels_np, preds_np, average='macro')
    return Auc,precision,recall,f1,z_samples,z_query,loss_val,{
        'loss': loss_val.item(),
        'acc': acc_val.item(),
        'y_hat': y_hat
        }
#-----------------Wavelet Decomposition Function--------------------------------
def Wavelet_conv(x_support,x_query,x_support1,x_query1):
  Supp1=[]
  for i,i1 in zip(x_support,x_support1):
    List_sup = []
    for j,j1 in zip(i,i1):
      #first level wavelet on original image
      LL1, (LH1, HL1, HH1) = pywt.dwt2(j1, 'bior3.1')
      SH=j.shape
      reshaped_LH1=np.resize(LH1, (SH[0],SH[0]))
      reshaped_LH1 = np.expand_dims(reshaped_LH1, axis=-1)
      Stacked=np.concatenate((reshaped_LH1,j), axis=2)
      List_sup.append(Stacked)
    Supp1.append(List_sup)
  Supp1=np.array(Supp1)
  Query1=[]
  for i,i1 in zip(x_query,x_query1):
    List_que = []
    for j,j1 in zip(i,i1):
      LL1, (LH1, HL1, HH1) = pywt.dwt2(j1, 'bior3.1')
      SH=j1.shape
      reshaped_LH1=np.resize(LH1, (SH[0],SH[0]))
      reshaped_LH1 = np.expand_dims(reshaped_LH1, axis=-1)
      Stacked=np.concatenate((reshaped_LH1,j), axis=2)
      List_que.append(Stacked)
    Query1.append(List_que)
  Query1=np.array(Query1)

  return Supp1,Query1
#-----------------Function to Calculate Prototpe--------------------------------
def CalcluatePrototype_Mean( z_samples ):
  z_proto=[]
  for i in z_samples:
    Class_proto=torch.mean(i, dim=0)
    z_proto.append(Class_proto)
  z_proto=torch.stack(z_proto)
  return z_proto
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
def train(M, optimizer, train_x, train_y,train_x1, train_y1,n_way, n_support, n_query, max_epoch, epoch_size,PATH_Model):
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
  Tr_Loss=[]
  Tr_Acc=[]
  Tr_Ep=[]
  while epoch < max_epoch and not stop:
    running_loss = 0.0
    running_acc = 0.0
    for episode in range(epoch_size):
      sample = extract_sample(n_way, n_support, n_query,train_x, train_y,train_x1, train_y1)
      optimizer.zero_grad()
      Auc,precision,recall,f1,z_samples,z_query,loss_val,output=M.set_forward_loss(sample,M)
      running_loss += output['loss']
      running_acc += output['acc']
      loss_val.backward()
      optimizer.step()
      #print('Epoch {:d} episode {:d}'.format(epoch,episode))
    epoch_loss = running_loss / epoch_size
    epoch_acc = running_acc / epoch_size
    Tr_Loss.append(epoch_loss)
    Tr_Acc.append(epoch_acc)
    Tr_Ep.append(epoch)
    print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc))

    epoch=epoch+1
    scheduler.step()
  EE1=datetime.datetime.now()
  print('***********Training Time******************:',EE1-EE)
  torch.save(M.state_dict(), PATH_Model)
  best_state = M.state_dict()
  return best_state,Tr_Loss,Tr_Acc,Tr_Ep
#---------------------------------Testing---------------------------------------
def test(M,test_x, test_y,test_x1, test_y1, n_way, n_support, n_query, test_episode):
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
  running_auc = 0.0
  running_pre = 0.0
  running_rec = 0.0
  running_f1 = 0.0
  Test_Acc_rec=[]
  for episode in range(test_episode):
    sample = extract_sample(n_way, n_support, n_query, test_x, test_y,test_x1, test_y1)
    Auc,precision,recall,f1,z_samples,z_query,loss_val,output=M.set_forward_loss(sample,M)
    Test_Acc_rec.append(output['acc'])
    running_loss += output['loss']
    running_acc += output['acc']
    running_auc += Auc
    running_pre += precision
    running_rec += recall
    running_f1 += f1
  print("*******************************************************")
  print('Testing ACC',running_acc/test_episode)
  print('Testing Auc',running_auc/test_episode)
  print('Testing Precision',running_pre/test_episode)
  print('Testing Recall',running_rec/test_episode)
  print('Testing F1',running_f1/test_episode)
  print("*******************************************************")
  return running_acc/test_episode,running_loss/test_episode,Test_Acc_rec
def Calc_Conf_Acc(Calc_Test_Conf):
  flattened_Calc_Test_Conf = Calc_Test_Conf

  mean_accuracy = np.mean(flattened_Calc_Test_Conf)

  std_dev = np.std(flattened_Calc_Test_Conf)

  # Calculate standard error of the mean
  std_error = stats.sem(flattened_Calc_Test_Conf)

  # Calculate margin of error for 95% confidence interval
  margin_of_error = stats.t.ppf(0.975, len(flattened_Calc_Test_Conf)-1) * std_error

  # Calculate lower and upper bounds of the confidence interval
  lower_bound = mean_accuracy - margin_of_error
  upper_bound = mean_accuracy + margin_of_error
  return mean_accuracy,margin_of_error
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------

#--------------------------------Main-------------------------------------------


def main():
  #--------------------------------Seed values------------------------------------
  Np_Seed1=int(sys.argv[1])
  Torch_Seed1=int(sys.argv[2])
  n_way =int(sys.argv[3])
  n_support =int(sys.argv[4])
  n_support1 =int(sys.argv[5])
  max_epoch =int(sys.argv[6])
  np.random.seed(Np_Seed1)
  torch.manual_seed(Torch_Seed1)
  Data='Pathology-random'
  #------------------------------------Pickling of Images-------------------------
  '''PATH='/home/ubuntu/RANJANA/Refined-Pickle-Files/Pickle-Wave-84_84/'
  trainx, trainy,trainx1, trainy1 = read_images('/home/ubuntu/RANJANA/Datasets/BreakHis-Data/Multi-Res/'+str(BC_res)+'X/Train-aug/')
  testx, testy,testx1, testy1 = read_images('/home/ubuntu/RANJANA/Datasets/BreakHis-Data/Multi-Res/'+str(BC_res)+'X/Test/')
  with open(PATH+Data+"-wave_gray_trainx.pkl","wb") as f:
    pickle.dump(trainx1,f)
  with open(PATH+Data+"-wave_gray_trainy.pkl","wb") as f:
    pickle.dump(trainy1,f)
  with open(PATH+Data+"-wave_gray_testx.pkl","wb") as f:
    pickle.dump(testx1,f)
  with open(PATH+Data+"-wave_gray_testy.pkl","wb") as f:
    pickle.dump(testy1,f)

  with open(PATH+Data+"-wave_trainx.pkl","wb") as f:
    pickle.dump(trainx,f)
  with open(PATH+Data+"-wave_trainy.pkl","wb") as f:
    pickle.dump(trainy,f)
  with open(PATH+Data+"-wave_testx.pkl","wb") as f:
    pickle.dump(testx,f)
  with open(PATH+Data+"-wave_testy.pkl","wb") as f:
    pickle.dump(testy,f)'''
  #--------------------------Parameters inside model------------------------------
  Dim_channel=4
  Actual_Hid_Dim=64
  Hid_Dim=65
  Hid_Dim1=66
  Hid_Dim2=67
  #---------------------------parameters to change--------------------------------

  n_query = 5
  epoch_size = 2000
  test_episode = 600

  PATH='/home/ranjana/Refined-Pickle-Files/Pickle-Wave-84_84/'
  PATH_Model='/home/ranjana/New_Models/Journal/WithSD/'+'Journal_CONV4_'+Data+'_'+'n_way_'+str(n_way)+'_n_support1_'+str(n_support1)+'.pth'

  #-------------------------------------------------------------------------------

  start=datetime.datetime.now()
  model = Proto_Conv4(in_channels=Dim_channel, out_channels=Actual_Hid_Dim,H_Dim=Hid_Dim,H_Dim1=Hid_Dim1,H_Dim2=Hid_Dim2).to('cuda')

  optimizer = optim.Adam(model.parameters(), lr = 0.001)

  with open(PATH+Data+"-wave_gray_trainx.pkl","rb") as f:
      trainx1 = pickle.load(f)
  with open(PATH+Data+"-wave_gray_trainy.pkl","rb") as f:
      trainy1 = pickle.load(f)
  with open(PATH+Data+"-wave_gray_testx.pkl","rb") as f:
      testx1 = pickle.load(f)
  with open(PATH+Data+"-wave_gray_testy.pkl","rb") as f:
      testy1 = pickle.load(f)

  with open(PATH+Data+"-wave_trainx.pkl","rb") as f:
      trainx = pickle.load(f)
  with open(PATH+Data+"-wave_trainy.pkl","rb") as f:
      trainy = pickle.load(f)
  with open(PATH+Data+"-wave_testx.pkl","rb") as f:
      testx = pickle.load(f)
  with open(PATH+Data+"-wave_testy.pkl","rb") as f:
      testy = pickle.load(f)

  print("*************************Training**********************************************")
  best_state,Tr_Loss,Tr_Acc,Tr_Ep=train(model, optimizer,trainx, trainy,trainx1, trainy1,n_way, n_support, n_query, max_epoch, epoch_size,PATH_Model)
  print("*********Training Loss*************",Tr_Loss)
  print("*********Training Accuracy*************",Tr_Acc)
  Tr_Acc_R = [x * 100 for x in Tr_Acc]
  plt.plot(Tr_Ep,Tr_Loss, label='Loss vs epoch')
  plt.plot(Tr_Ep,Tr_Acc_R, label='Accuracy vs epoch')
  plt.title('Plot of Loss and Accuracy vs epoch')
  plt.legend()
  # Display the plot
  plt.show()
  print("*************************Testing**********************************************")
  model.load_state_dict(best_state)
  model.eval()

  T_Acc,T_Loss,Test_Acc_rec=test(model,testx, testy,testx1, testy1,n_way, n_support1, n_query,test_episode)
  end=datetime.datetime.now()
  #---------calculate average accuracy with 95% confidence interval-----------------------
  mean_accuracy,margin_of_error=Calc_Conf_Acc(Test_Acc_rec)
  print("^^^^^^^^^^^^^^Average Accuracy:^^^^^^^^^^^^^^^^^^^^", mean_accuracy)
  print("^^^^^^^^^^^^^^^95% Confidence Interval Error margin: [{:.4f}]^^^^^^^^^^^^^^^^^^".format(margin_of_error))
  print("total time taken:",end-start)
  num_params = count_parameters(model)
  print(f"Number of parameters in the model: {num_params}")
if __name__ == "__main__":
    main()
