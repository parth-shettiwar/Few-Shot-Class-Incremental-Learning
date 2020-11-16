# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_distance, extract_features_classification
from evaluations import Recall_at_ks, NMI, Recall_at_ks_products
# import DataSet
import os
from torch.autograd import Variable

import numpy as np
from utils import to_numpy
# import pdb
from torch.nn import functional as F
import torchvision.transforms as transforms
from ImageFolder import *
from utils import *
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
from CIFAR100 import CIFAR100
from models.resnet import Generator, Discriminator,ClassifierMLP,ModelCNN
cudnn.benchmark = True
from copy import deepcopy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
# from tensorboardX import SummaryWriter
# writer = SummaryWriter('logs')
cudnn.benchmark = True
from copy import deepcopy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='cifar100')
parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH')
parser.add_argument('-name', type=str, default='tmp', metavar='PATH')

parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
parser.add_argument('-seed', default=1993, type=int, metavar='N',
                    help='seeds for training process')
parser.add_argument('-epochs', default=600, type=int, metavar='N', help='epochs for training process')
parser.add_argument('-num_task', type=int, default=3, help="learning rate of new parameters")
parser.add_argument('-nb_cl_fg', type=int, default=4, help="learning rate of new parameters")

parser.add_argument('-num_class', type=int, default=10, help="learning rate of new parameters")
parser.add_argument('-dir', default='/data/datasets/featureGeneration/',
                        help='data dir')
parser.add_argument('-top5', action = 'store_true', help='output top5 accuracy')

args = parser.parse_args()
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# if 'cifar' in args.data:
#     transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#     testdir = args.dir + '/cifar'

# if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
#         mean_values = [0.485, 0.456, 0.406]
#         std_values = [0.229, 0.224, 0.225]
#         transform_test = transforms.Compose([
#             # transforms.Resize(224),
#             transforms.CenterCrop(224),
#             # transforms.RandomResizedCrop(224),
#             # transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean_values,
#                                  std=std_values)
#         ])
#         # traindir = os.path.join(args.dir, 'tiny-imagenet-200', 'train')
#         testdir = "/content/drive/My Drive/btp/GFR-IL-master/dataset/imagenette2/val"
num_classes = args.num_class
num_task = args.num_task
num_class_per_task = (num_classes -  args.nb_cl_fg) // num_task
np.random.seed(args.seed)
random_perm = list(range(num_classes))
print('Test starting -->\t')
index = random_perm[:args.nb_cl_fg + args.num_task * num_class_per_task]
print(index,"index")
# if 'imagenet' in args.data:
#     dataset = ImageFolder(testdir, transform_test, index=index,num_instance_per_class=20)
#     print(dataset)
#     # dataset = ImageFolder(testdir, transform_test, index=index,num_instance_per_class=20)
#     # testfolder = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), 10, replace=False))
#     test_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=10,
#         shuffle=False,
#         drop_last=False, num_workers=4)
# elif args.data =='cifar100':
#     np.random.seed(args.seed)
#     target_transform = np.random.permutation(num_classes)
#     testset = CIFAR100(root=testdir, train=False, download=True, transform=transform_test, target_transform = target_transform, index = index)
#     train_loader = torch.utils.data.DataLoader(
#         trainfolder, batch_size=args.BatchSize,
#         shuffle=False,
#         drop_last=True, num_workers=args.nThreads)
model = torch.load("/content/drive/My Drive/btp/GFR-IL-master/checkpoints/task_03_9999_model_generator.pkl")
model =model.to(device)
print(model)
import models
model2 = models.create('resnet18', pretrained=True, feat_dim=512,embed_dim=512)
model2 =model2.to(device)

# embeddings1 = []
# embeddings_labels1 = []
# with torch.no_grad():
#         # print(len(data_loader),"^^^^^^^^^^^^^")
#   print(len(test_loader))      
#   for i, data in enumerate(test_loader, 1):
#       inputs, labels = data
#       if(i==0):
#         print(inputs[0])
#         print(labels)
#       inputs = Variable(inputs.to(device))
#       embed_feat = model2(inputs)
#       print(labels)
#       embeddings_labels1.append(labels.numpy())
#       embeddings1.append(embed_feat.cpu().numpy())      
# for i in range(1):
# embeddings1  = np.asarray(embeddings1,dtype=np.float64)
# embeddings1 = np.reshape(embeddings1, (embeddings1.shape[0] * embeddings1.shape[1], embeddings1.shape[2]))
# embeddings_labels1 = np.asarray(embeddings_labels1)
# print("embdddddddddd",embeddings1[0])
# print("embdddddddddd",embeddings1[20])
# print("embdddddddddd",embeddings1[40])
    
# embeddings_labels1 = np.reshape(embeddings_labels1, embeddings_labels1.shape[0] * embeddings_labels1.shape[1])
# print(embeddings1.shape)
num_class = args.num_class
final_prot = np.zeros((1,10,512))

prots = np.loadtxt("/content/drive/My Drive/btp/GFR-IL-master/results/prots.txt")
embeddings1 = np.loadtxt("/content/drive/My Drive/btp/GFR-IL-master/results/data.txt")
labs = np.loadtxt("/content/drive/My Drive/btp/GFR-IL-master/results/data_lab.txt")
num = 50
sigma = .1
print(prots)
syn_label_old = torch.zeros(num_class,num_class).to(device)
print("oldddddddddddddddd",syn_label_old)
for i in range(num_class):
  syn_label_old[i][i] = 1
for i in range(num):

  z = torch.Tensor(np.random.normal(0, sigma, (num_class, 512))).to(device)
  # print(z)
  # print("hhhhhhhhhhhhhhh")
  # print((model(z, syn_label_old)))
  final_prot = np.concatenate((final_prot, np.expand_dims((model(z, syn_label_old)).cpu().detach().numpy(),axis=0)), axis=0)
print(final_prot.shape)
# z = torch.Tensor(np.random.normal(0, 1, (len(prots["class_mean"]), args.feat_dim))).to(device)
# final_prot = np.concatenate((final_prot, (model(z, (torch.FloatTensor(prots["class_mean"]).to(device)))+z).cpu().detach().numpy()), axis=0)      
final_prot = final_prot[1:] 
print(final_prot.shape)  
print(torch.from_numpy(embeddings1).float())
log_dir = "/content/drive/My Drive/btp/GFR-IL-master/checkpoints/imagenet_sub_10tasks"
# cd1 = torch.cdist(torch.from_numpy(prots).float(),torch.from_numpy(embeddings1).float(),p=2)
cd2 = torch.zeros((2000,num,10))
for i in range(num):
  cd2[:,i,:] = torch.cdist(torch.from_numpy(embeddings1).float(),torch.from_numpy(final_prot[i]).float(),p=2)
# print(torch.argmin(cd2,dim=0).numpy())  
cd2 = cd2.numpy()
kk = np.zeros((2000))
for i in range(2000):
  a,b = np.where(cd2[i,:,:] == np.amin(cd2[i,:,:]))
  kk[i] = b
# print(torch.argmin(cd1,dim=0))    
print(kk)
y_pred = kk
# y_pred = torch.argmin(cd2,dim=0).numpy()  
# print(se.shape)
print(labs.shape) 
import itertools
import numpy as np
from os import path

import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
labs = np.zeros((200))
for i in range(9):
  labs = np.concatenate((labs,(i+1)*np.ones((200))),axis=0)

# labs = np.concatenate(labs,2*np.ones(200))
# labs = np.concatenate(labs,3*np.ones(200))
print(y_pred)
sum1=0
conf_mat = confusion_matrix(y_pred, labs)
acc = np.zeros((num_class))
for i in range(num_class):

  acc[i] = conf_mat[i][i] / np.sum(conf_mat[:,i])
  sum1 =sum1 + conf_mat[i][i]
acc2 = sum1/np.sum(conf_mat)
print("                           ")
print("                           ")
print("                           ")
print("Overall Accuracy" ,acc2*100)
print("Class Wise Accuracy",acc*100)
print("                           ")
print("                           ")
print("                           ")

print(conf_mat)
if(path.exists("/content/drive/My Drive/btp/GFR-IL-master/results/pseudo.txt")):
  os.remove("/content/drive/My Drive/btp/GFR-IL-master/results/pseudo.txt")
file1 = open("/content/drive/My Drive/btp/GFR-IL-master/results/pseudo.txt","x")
final_prot = np.reshape(final_prot, (final_prot.shape[0]*final_prot.shape[1],final_prot.shape[2]))
np.savetxt(file1,final_prot)