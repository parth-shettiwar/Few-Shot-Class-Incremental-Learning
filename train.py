# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import getpass
import os
from tqdm import tqdm
import sys
import torch
import torch.utils.data
import pdb
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.autograd import Variable
import models

from utils import RandomIdentitySampler, mkdir_if_missing, logging, display,truncated_z_sample
from torch.optim.lr_scheduler import StepLR
import numpy as np
from ImageFolder import *
import torch.utils.data	
import torch.nn.functional as F
import torchvision.transforms as transforms
from evaluations import extract_features, pairwise_distance
from models.resnet import Generator, Discriminator,ClassifierMLP,ModelCNN
import torch.autograd as autograd
import scipy.io as sio
from CIFAR100 import CIFAR100

pa = "/content/drive/My Drive/btp/GFR-IL-master/checkpoints"
cudnn.benchmark = True
from copy import deepcopy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
sigma = .1
def to_binary(labels,args):
    # Y_onehot is used to generate one-hot encoding
    y_onehot = torch.FloatTensor(len(labels), args.num_class)
    y_onehot.zero_()
    y_onehot.scatter_(1, labels.cpu()[:,None], 1)
    code_binary = y_onehot.to(device)
    return code_binary

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def compute_gradient_penalty(D, real_samples, fake_samples, syn_label):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    d_interpolates, _ = D(interpolates, syn_label)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = \
        autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True,
                      retain_graph=True,
                      only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


def compute_prototype(model, data_loader,current_task,num_class_per_task):
    model.eval()
    # print("HHHHHH")
    # print("len",len(data_loader))
    count = 0
    embeddings1 = []
    embeddings_labels1 = []
    embeddings2 = []
    embeddings_labels2 = []
    # x = len(data_loader)
    # if(current_task==0):
    #   bb = x/args.nb_cl_fg
    #   ter = 30
    #   # ter = ter/args.nb_cl_fg
    # else:
    #   bb = x/num_class_per_task
    #   ter = 30
    #   # ter = ter/num_class_per_task 
    # print(ter)
    # print(bb)
    # print(len(data_loader))
    with torch.no_grad():
        # print(len(data_loader),"^^^^^^^^^^^^^")
        for i, data in enumerate(data_loader, 0):
            # if i>1:
                # break
            
            inputs, labels = data
            inputs = Variable(inputs.to(device))
            embed_feat = model(inputs)
            # if(count==bb):
              # count=0
            # if(count<ter):

            embeddings_labels1.append(labels.numpy())
            embeddings1.append(embed_feat.cpu().numpy())
            # else:

              # embeddings_labels2.append(labels.numpy())
              # embeddings2.append(embed_feat.cpu().numpy())  
            # count += 1
    embeddings1 = np.asarray(embeddings1)
    embeddings1 = np.reshape(embeddings1, (embeddings1.shape[0] * embeddings1.shape[1], embeddings1.shape[2]))
    embeddings_labels1 = np.asarray(embeddings_labels1)
    # print(embeddings1[0])
    print("embdddddddddd",embeddings_labels1)
    # print("embdddddddddd",embeddings_labels2)
    embeddings_labels1 = np.reshape(embeddings_labels1, embeddings_labels1.shape[0] * embeddings_labels1.shape[1])
    # embeddings2 = np.asarray(embeddings2)
    # print(embeddings2.shape,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # embeddings2 = np.reshape(embeddings2, (embeddings2.shape[0] * embeddings2.shape[1], embeddings2.shape[2]))
    # embeddings_labels2 = np.asarray(embeddings_labels2)
    # print("embdddddddddd",embeddings_labels2)
    
    # embeddings_labels2 = np.reshape(embeddings_labels2, embeddings_labels2.shape[0] * embeddings_labels2.shape[1])
    labels_set = np.unique(embeddings_labels1)
    
    class_mean = []
    class_std = []
    class_label = []
    embeddings_final = np.zeros((1,512))
    embeddings_final2 = np.zeros((1,512))
    for i in labels_set:
        ind_cl1 = np.where(i == embeddings_labels1)[0]
        # ind_cl2 = np.where(i == embeddings_labels2)[0]
        embeddings_tmp = embeddings1[ind_cl1]
        # embeddings_tmp2 = embeddings2[ind_cl2]
        class_label.append(i)
        class_mean.append(np.mean(embeddings_tmp, axis=0))
        class_std.append(np.std(embeddings_tmp, axis=0))
        embeddings_final = np.concatenate((embeddings_final,embeddings_tmp),axis=0)
        # embeddings_final2 = np.concatenate((embeddings_final2,embeddings_tmp2),axis=0)
    # print("means^^^^^^^^^^^^^^^^^^^^^^^^^",class_mean)    
    prototype = {'class_mean': class_mean, 'class_std': class_std,'class_label': class_label}
    # print(len(class_mean))
    # print(len(class_mean[0])
    # print(embeddings_tmp)
    # print(np.where(embeddings_labels==))
    embeddings_labels2 = np.zeros((1))
    embeddings_final2 = np.zeros((1,512))
    return prototype,embeddings_final[1:],embeddings_final2[1:],embeddings_labels2[1:]


def train_task(args, train_loader, current_task, prototype={}, pre_index=0):
    num_class_per_task = (args.num_class-args.nb_cl_fg) // args.num_task
    # print(num_class_per_task)
    # print(args.num_class)
    # print(args.nb_cl_fg)
    # print(args.num_task)
    task_range = list(range(args.nb_cl_fg + (current_task - 1) * num_class_per_task, args.nb_cl_fg + current_task * num_class_per_task))
    if num_class_per_task==0:
        pass  # JT
    else:
        old_task_factor = args.nb_cl_fg // num_class_per_task + current_task - 1
    # print("&&&&&&&&&&&&&&&&&&&",old_task_factor)
    log_dir = os.path.join(args.ckpt_dir, args.log_dir)
    mkdir_if_missing(log_dir)

    sys.stdout = logging.Logger(os.path.join(log_dir, 'log_task{}.txt'.format(current_task)))
    tb_writer = SummaryWriter(log_dir)
    display(args)
    # print("$$$$$$$$$$$$$$")
    # One-hot encoding or attribute encoding
    if 'imagenet' in args.data:
        import models
        model = models.create('resnet18', pretrained=True, feat_dim=args.feat_dim,embed_dim=512)
        # print(model)    
    elif 'cifar' in args.data:
        model = models.create('resnet18_cifar', pretrained=False, feat_dim=args.feat_dim,embed_dim=args.num_class)

    # print("^^^^^^^^^^^^^^^^^^")
    # if current_task > 0:
    #     model = torch.load(os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model.pkl' % int(args.epochs - 1)))
    #     model_old = deepcopy(model)
    #     model_old.eval()
    #     model_old = freeze_model(model_old)

    model = model.to(device)
       
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

    loss_mse = torch.nn.MSELoss(reduction='sum')
    
    # Loss weight for gradient penalty used in W-GAN
    lambda_gp = args.lambda_gp
    lambda_lwf = args.gan_tradeoff
    # Initialize generator and discriminator
    if current_task == 0:
        generator = Generator(feat_dim=args.feat_dim,latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, class_dim=args.num_class,num_class =  args.num_class)
        discriminator = Discriminator(feat_dim=args.feat_dim,hidden_dim=args.hidden_dim, class_dim=args.num_class,latent_dim=args.latent_dim)
    else:
        generator = torch.load(os.path.join(pa, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_generator.pkl' % int(args.epochs_gan - 1)))
        discriminator = torch.load(os.path.join(pa, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_discriminator.pkl' % int(args.epochs_gan - 1)))
        generator_old = deepcopy(generator)
        generator_old.eval()
        generator_old = freeze_model(generator_old)

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    # print("kkkk")
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    scheduler_G = StepLR(optimizer_G, step_size=200, gamma=0.3)
    scheduler_D = StepLR(optimizer_D, step_size=200, gamma=0.3)
    
    # Y_onehot is used to generate one-hot encoding
    y_onehot = torch.FloatTensor(args.BatchSize, args.num_class)

    for p in generator.parameters():  # set requires_grad to False
        p.requires_grad = False  


    import torchvision.models as models
    # print("ggg")
    # mod = torchvision.models.resnet50(pretrained=True, progress=True, **kwargs)

    # torchvision.models.resnet50(pretrained=False, progress=True, **kwargs)    

    

    prots,embeddings_tmp,embeddings_tmp2,embeddings_labels2 = compute_prototype(model,train_loader,current_task,num_class_per_task)
    classes_num = len(prots["class_mean"])
    # print("mmm")
    # print(len(prots["class_mean"]))
    # print("WGAN")
    model = model.eval()
    # for p in model.parameters():  # set requires_grad to False
    #     p.requires_grad = False
    for p in generator.parameters():  # set requires_grad to False
        p.requires_grad = True
    criterion_softmax = torch.nn.CrossEntropyLoss().to(device)
    # if current_task != args.num_task:
    for epoch in range(args.epochs_gan):
        loss_log = {'D/loss': 0.0,
                    'D/new_rf': 0.0,
                    'D/new_lbls': 0.0,
                    'D/new_gp': 0.0,
                    'D/prev_rf': 0.0,
                    'D/prev_lbls': 0.0,
                    'D/prev_gp': 0.0,
                    'G/loss': 0.0,
                    'G/new_rf': 0.0,
                    'G/new_lbls': 0.0,
                    'G/prev_rf': 0.0,
                    'G/prev_mse': 0.0,
                    'G/new_classifier':0.0,
                    'E/kld': 0.0,
                    'E/mse': 0.0,
                    'E/loss': 0.0}
        scheduler_D.step()
        scheduler_G.step()
        # for i, data in enumerate(train_loader, 0):
        # for i in range(len(prots["class_mean"])):
        for p in discriminator.parameters():
            p.requires_grad = True
        # inputs, labels = dat
        # inputs = Variable(inputs.to(device)
        ############################# Train Disciminator###########################
        optimizer_D.zero_grad()
        # real_feat = model(inputs)
        z = torch.Tensor(np.random.normal(0, sigma, (classes_num, args.feat_dim))).to(device)                
        z1 = torch.Tensor(np.random.normal(0, sigma, (classes_num, args.feat_dim))).to(device)                
        z2 = torch.Tensor(np.random.normal(0, sigma, (classes_num, args.feat_dim))).to(device)               
        # y_onehot.zero_()
        # y_onehot.scatter_(1, labels[:, None], 1)
        # syn_label = y_onehot.to(device)
        syn_label = torch.zeros(classes_num,args.num_class).to(device) 
        if(current_task == 0):
          for i in range(classes_num):
            syn_label[i][i] = 1 
        else:
          for i in range(classes_num):
            syn_label[i][(args.nb_cl_fg) + (current_task-1)*num_class_per_task+i] = 1 
        print("SYNNNNNNNNNNNNNNNNNNN",syn_label)  
        # print("shhhhh",torch.FloatTensor(prots["class_mean"]).shape)
        fake_feat = generator(z, syn_label)
        # print("fake",fake_feat.shape)
        fake_validity, _               = discriminator(fake_feat+z1,syn_label)
        # print("valid",fake_validity.shape)
        real_validity, disc_real_acgan = discriminator(torch.FloatTensor(prots["class_mean"]).to(device)+z2,syn_label)
        # print("second",real_validity.shape)
        # print("faekkkkkkkkkkk",fake_validity)
        # print("reallllllllll",real_validity)
        # Adversarial loss
        d_loss_rf = -torch.mean(real_validity) + torch.mean(fake_validity)
        gradient_penalty = compute_gradient_penalty(discriminator, torch.FloatTensor(prots["class_mean"]).to(device), fake_feat.to(device), syn_label.to(device)).mean()
        # gradient_penalty = 0
        # d_loss_lbls = criterion_softmax(disc_real_acgan, labels.to(device))
        d_loss = d_loss_rf + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()
        loss_log['D/loss'] += d_loss
        loss_log['D/new_rf'] += d_loss_rf
        loss_log['D/new_lbls'] += 0 #!!!
        # loss_log['D/new_gp'] += gradient_penalty.item() if lambda_gp != 0 else 0
        del d_loss_rf
        # print("GENERATOR")
        
        ############################# Train Generaator###########################
        # Train the generator every n_critic steps
        if i % args.n_critic == 0 or True:
            for p in discriminator.parameters():
                p.requires_grad = False                   
            ############################# Train GAN###########################
            optimizer_G.zero_grad()
            # Generate a batch of images
            fake_feat = generator(z, syn_label)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity, disc_fake_acgan = discriminator(fake_feat+z1, syn_label)
            if current_task == 0:
                loss_aug = 0 * torch.sum(fake_validity)
                # print("kejwhfkjewhfkjew")
            else:
                # print("heheeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                # ind = list(range(len(pre_index)))
                # embed_label_sythesis = []
                # for _ in range(args.BatchSize):
                #     np.random.shuffle(ind)
                #     embed_label_sythesis.append(pre_index[ind[0]])
                # embed_label_sythesis = np.asarray(embed_label_sythesis)
                # embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
                # y_onehot.zero_()
                # y_onehot.scatter_(1, embed_label_sythesis[:, None], 1)
                # syn_label_pre = y_onehot.to(device)
                oldc = len(prototype["class_mean"])
                syn_label_old = torch.zeros(oldc,args.num_class).to(device)
                
                for i in range(oldc):
                  syn_label_old[i][i] = 1
                print("oldddddddddddddddd",syn_label_old)  
                z3 = torch.Tensor(np.random.normal(0, sigma, (oldc, args.feat_dim))).to(device)                
                pre_feat = generator(z3, syn_label_old) 
                pre_feat_old = generator_old(z3, syn_label_old)
                loss_aug = loss_mse(pre_feat, pre_feat_old)
            mse_curr = loss_mse(fake_feat, torch.FloatTensor(prots["class_mean"]).to(device))    
            g_loss_rf = -torch.mean(fake_validity)
            # g_loss_lbls = criterionsoftmax(disc_fake_acgan, labels.to(device))
            g_loss = g_loss_rf + current_task*old_task_factor * loss_aug + mse_curr
            # g_loss = g_loss_rf  
            # print()
            print("mse",mse_curr)
            print("generator lossssssss",g_loss)
            print("prev loss", loss_aug)
            print("task",old_task_factor)
            print("ref",g_loss_rf)
            loss_log['G/loss'] += g_loss.item()
            loss_log['G/new_rf'] += g_loss_rf.item()
            loss_log['G/new_lbls'] += 0 #!
            loss_log['G/new_classifier'] += 0 #!
            loss_log['G/prev_mse'] += loss_aug.item() if lambda_lwf != 0 else 0
            del g_loss_rf
            g_loss.backward()
            optimizer_G.step()
        # print('[GAN Epoch %05d]\t D Loss: %.3f \t G Loss: %.3f \t LwF Loss: %.3f' % (
            # epoch + 1, loss_log['D/loss'], loss_log['G/loss'], loss_log['G/prev_rf']))
        for k, v in loss_log.items():
            if v != 0:
                tb_writer.add_scalar('Task {} - GAN/{}'.format(current_task, k), v, epoch + 1)
        if epoch ==args.epochs_gan - 1:
            
            print(os.path.join(pa, 'task_' + str(current_task).zfill(2) + '_%d_model_generator.pkl' % epoch))
            torch.save(generator, os.path.join(pa, 'task_' + str(
                current_task).zfill(2) + '_%d_model_generator.pkl' % epoch))
            print("(((((((((###########)))))))))")    
            torch.save(discriminator, os.path.join(pa, 'task_' + str(
                current_task).zfill(2) + '_%d_model_discriminator.pkl' % epoch))
    tb_writer.close()
    final_prot = np.zeros((1,args.feat_dim))
    # print("atgs",args.num_class)
    # print("curr",current_task)
    if(current_task==args.num_task):
      for i in range(1):
        z = torch.Tensor(np.random.normal(0, sigma, (oldc, args.feat_dim))).to(device)
        final_prot = np.concatenate((final_prot, (generator(z, syn_label_old)+z).cpu().detach().numpy()), axis=0)
        z = torch.Tensor(np.random.normal(0, sigma, (len(prots["class_mean"]), args.feat_dim))).to(device)
        final_prot = np.concatenate((final_prot, (generator(z, syn_label)+z).cpu().detach().numpy()), axis=0)
    # prototype = compute_prototype(model,train_loader)  #!
    return prots,final_prot[1:,:],embeddings_tmp,embeddings_tmp2,embeddings_labels2


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generative Feature Replay Training')


    # task setting
    parser.add_argument('-data', default='cifar100', required=True, help='path to Data Set')
    parser.add_argument('-num_class', default=10, type=int, metavar='n', help='dimension of embedding space')
    parser.add_argument('-nb_cl_fg', type=int, default=4, help="Number of class, first group")
    parser.add_argument('-num_task', type=int, default=3, help="Number of Task after initial Task")

    # method parameters
    parser.add_argument('-mean_replay', action = 'store_true', help='Mean Replay')
    parser.add_argument('-tradeoff', type=float, default=1.0, help="Feature Distillation Loss")

    # basic parameters
    parser.add_argument('-load_dir_aug', default='', help='Load first task')
    parser.add_argument('-ckpt_dir', default='checkpoints', help='checkpoints dir')
    parser.add_argument('-dir', default='/data/datasets/featureGeneration/', help='data dir')
    parser.add_argument('-log_dir', default="/content/drive/My Drive/btp/GFR-IL-master/checkpoints", help='where the trained models save')
    parser.add_argument('-name', type=str, default='tmp', metavar='PATH')

    parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
    parser.add_argument('-nThreads', '-j', default=4, type=int, metavar='N', help='number of data loading threads')

    # hyper-parameters
    parser.add_argument('-BatchSize', '-b', default=5, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-lr', type=float, default=1e-3, help="learning rate of new parameters")
    parser.add_argument('-lr_decay', type=float, default=0.1, help='Decay learning rate')
    parser.add_argument('-lr_decay_step', type=float, default=100, help='Decay learning rate every x steps')
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-weight-decay', type=float, default=2e-4)

    # hype-parameters for W-GAN
    parser.add_argument('-gan_tradeoff', type=float, default=1e-3, help="learning rate of new parameters")
    parser.add_argument('-gan_lr', type=float, default=1e-4, help="learning rate of new parameters")
    parser.add_argument('-lambda_gp', type=float, default=10.0, help="learning rate of new parameters")
    parser.add_argument('-n_critic', type=int, default=5, help="learning rate of new parameters")

    parser.add_argument('-latent_dim', type=int, default=200, help="learning rate of new parameters")
    parser.add_argument('-feat_dim', type=int, default=512, help="learning rate of new parameters")
    parser.add_argument('-hidden_dim', type=int, default=512, help="learning rate of new parameters")
    
    # training parameters
    parser.add_argument('-epochs', default=2, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-epochs_gan', default=1001, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-seed', default=1993, type=int, metavar='N', help='seeds for training process')
    parser.add_argument('-start', default=0, type=int, help='resume epoch')

    args = parser.parse_args()

    # Data
    # print('==> Preparing data..')
    
    if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            # transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        # traindir = os.path.join(args.dir, 'tiny-imagenet-200', 'train')
        traindir = "/content/drive/My Drive/btp/GFR-IL-master/dataset/imagenette2/train"
    if  args.data == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        traindir = args.dir + '/cifar'

    num_classes = args.num_class 
    num_task = args.num_task
    num_class_per_task = (num_classes-args.nb_cl_fg) // num_task
    
    random_perm = list(range(num_classes))      # multihead fails if random permutaion here
    # print("rrrrrrrrrrrrrrrrrr",random_perm)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    prototype = {}
    embeddings_tmp=np.zeros((1,512))
    embeddings_tmp2=np.zeros((1,512))
    embeddings_labels2=np.zeros((1))

    if args.mean_replay:
        args.epochs_gan = 2
        
    for i in range(args.start, num_task+1):
        # print ("-------------------Get started--------------- ")
        # print ("Training on Task " + str(i))
        if i == 0:
            pre_index = 0
            class_index = random_perm[:args.nb_cl_fg]
            # print("classsssssssssssssssss",class_index)
        else:
            pre_index = random_perm[:args.nb_cl_fg + (i-1) * num_class_per_task]
            class_index = random_perm[args.nb_cl_fg + (i-1) * num_class_per_task:args.nb_cl_fg + (i) * num_class_per_task]
        if args.data == 'cifar100':

            np.random.seed(args.seed)
            target_transform = np.random.permutation(num_classes)
            trainset = CIFAR100(root=traindir, train=True, download=True, transform=transform_train, target_transform = target_transform, index = class_index)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.BatchSize, shuffle=True, num_workers=args.nThreads,drop_last=True)
        else:
          if(i==0):
            numc = 100
          else:  
            numc = 5
          print("%%%%%",class_index)  
          trainfolder = ImageFolder(traindir, transform_train, index=class_index,num_instance_per_class=numc)
          train_loader = torch.utils.data.DataLoader(
              trainfolder, batch_size=args.BatchSize,
              shuffle=False,
              drop_last=True, num_workers=args.nThreads)

        prototype_old = prototype
        
        prototype,final,temp,temp2,labels_temp2 = train_task(args, train_loader, i, prototype=prototype, pre_index=pre_index)
        
        embeddings_tmp = np.concatenate((embeddings_tmp,temp),axis=0)
        embeddings_tmp2 = np.concatenate((embeddings_tmp2,temp2),axis=0)
        embeddings_labels2 = np.concatenate((embeddings_labels2,labels_temp2),axis=0)
        # print("&&&&&&&&&&&&&",embeddings_tmp2.shape)
        if args.start>0:
            pass
        else:
            if i >= 1:
                # Increase the prototype as increasing number of task
                for k in prototype.keys():
                    prototype[k] = np.concatenate(( prototype_old[k],prototype[k]), axis=0)
                    
        # print("priotiititititit",len(prototype["class_mean"]))
        # if(i==0):
        #   break
  
    # print(prototype)
    # print(len(embeddings_tmp))
    # print(len(embeddings_tmp[0]))
    embeddings_tmp = np.asarray(embeddings_tmp,dtype=np.float64)
    embeddings_tmp2 = np.asarray(embeddings_tmp2,dtype=np.float64)
    prototype["class_mean"] = np.asarray(prototype["class_mean"],dtype=np.float64)
    ccc = torch.cdist(torch.from_numpy(prototype["class_mean"]).float(),torch.from_numpy(embeddings_tmp).float(),p=2)
    ccc2 = torch.cdist(torch.from_numpy(prototype["class_mean"]).float(),torch.from_numpy(embeddings_tmp2).float(),p=2)
    print(ccc.shape)
    print(torch.argmin(ccc,dim=0))
    print(torch.argmin(ccc2,dim=0))
    import os.path
    from os import path
    prototypen = np.asarray(prototype["class_mean"], dtype=np.float64)
    print(len(final))
    print(len(final[0]))
    final = np.asarray(final, dtype=np.float64)
    cd = torch.cdist(torch.from_numpy(final).float(),torch.from_numpy(embeddings_tmp).float(),p=2)
    cd2 = torch.cdist(torch.from_numpy(final).float(),torch.from_numpy(embeddings_tmp2).float(),p=2)

    print(torch.argmin(cd,dim=0))
    print(torch.argmin(cd2,dim=0))
    print(final.shape)
    # print(prototypen)
    if(path.exists("/content/drive/My Drive/btp/GFR-IL-master/results/prots.txt")):
      os.remove("/content/drive/My Drive/btp/GFR-IL-master/results/prots.txt")
    # if(path.exists("/content/drive/My Drive/btp/GFR-IL-master/results/data.txt")):
      # os.remove("/content/drive/My Drive/btp/GFR-IL-master/results/data.txt") 
    # if(path.exists("/content/drive/My Drive/btp/GFR-IL-master/results/data_lab.txt")):
      # os.remove("/content/drive/My Drive/btp/GFR-IL-master/results/data_lab.txt")   
    file1 = open("/content/drive/My Drive/btp/GFR-IL-master/results/prots.txt","x")  
    # file2 = open("/content/drive/My Drive/btp/GFR-IL-master/results/data.txt","x")
    # file3 = open("/content/drive/My Drive/btp/GFR-IL-master/results/data_lab.txt","x")
    np.savetxt(file1, prototypen)
    # np.savetxt(file2, embeddings_tmp2[1:])
    # np.savetxt(file3, embeddings_labels2[1:])
    # print(embeddings_labels2.shape)
    # print(embeddings_tmp2.shape)
    # print(embeddings_tmp.shape)
# for i in output['embedding']:
#   store = i.cpu().detach().numpy()
#   print(store.shape)
  
#   coun1=coun1+1
# print(coun1)





