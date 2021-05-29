import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision
import numpy as np
import task_generator as tg
import os, sys
import math
import argparse
import scipy as sp
import scipy.stats
import time

sys.path.append('../arc')
import models

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 15)
parser.add_argument("-s","--support_num_per_class",type = int, default = 1)
parser.add_argument("-q","--query_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-sigma","--sigma", type = float, default = 150)
parser.add_argument("-beta","--beta", type = float, default = 0.01)
args = parser.parse_args()


# Hyper Parameters
METHOD = "SMH_VAE_NN_" + str(args.beta)
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SUPPORT_NUM_PER_CLASS = args.support_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
SIGMA = args.sigma
BETA2 = args.beta

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits
    
def power_norm(x, SIGMA):
	out = 2*F.sigmoid(SIGMA*x) - 1
	return out
	
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    print("init data folders")
    metatrain_folders,metaquery_folders = tg.mini_imagenet_folders()

    print("init neural networks")
    foreground_encoder = models.VAEEncoder().apply(weights_init).cuda()
    background_encoder = models.VAEEncoder().apply(weights_init).cuda()
    vae_decoder = models.VAEDecoder().apply(weights_init).cuda()
    mixture_network = models.MixtureNetwork(FEATURE_DIM).apply(weights_init).cuda()
    relation_network = models.SimilarityNetwork(FEATURE_DIM,RELATION_DIM).apply(weights_init).cuda()

    optimizer = torch.optim.Adam([{'params': foreground_encoder.parameters()}, 
                                 {'params': background_encoder.parameters()},
                                 {'params': vae_decoder.parameters()}], lr=LEARNING_RATE)
                                 
    optimizer_scheduler = StepLR(optimizer,step_size=100000,gamma=0.5)
	
	# Loading models
    if os.path.exists("checkpoints/" + METHOD + "/checkpoint_up_to_date.pth.tar"):
    	checkpoint = torch.load("checkpoints/" + METHOD + "/checkpoint_up_to_date.pth.tar")
    	foreground_encoder.load_state_dict(checkpoint['foreground_encoder'])
    	background_encoder.load_state_dict(checkpoint['background_encoder'])
    	vae_decoder.load_state_dict(checkpoint['vae_decoder'])
    	optimizer.load_state_dict(checkpoint['optimizer'])
    	print("load modules successfully!")
     
    if os.path.exists("checkpoints/" + str(METHOD)) == False:
        os.system("mkdir checkpoints/" + str(METHOD) )

    print("Start Training...")

    mse = nn.MSELoss().cuda()
    ce = nn.CrossEntropyLoss().cuda()
    best_accuracy = 0.0
    checkpoint = {}
    checkpoint['acc'] = []        
    for episode in range(EPISODE):   
        optimizer_scheduler.step(episode)

        # init dataset
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SUPPORT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
        support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SUPPORT_NUM_PER_CLASS,split="train",shuffle=False)
        query_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=QUERY_NUM_PER_CLASS,split="test",shuffle=True)

        # support datas
        support_img,support_sal,support_labels = support_dataloader.__iter__().next()
        query_img,query_sal,query_labels = query_dataloader.__iter__().next()
        
        # calculate features      
        supp_fore_mu, supp_fore_sigma = foreground_encoder(Variable(support_img*support_sal).cuda()) #obtain foreground features
        supp_back_mu, supp_back_sigma = background_encoder(Variable(support_img*(1-support_sal)).cuda()) #obtain background features
        query_fore_mu, query_fore_sigma = foreground_encoder(Variable(query_img*query_sal).cuda()) #obtain foreground features
        query_back_mu, query_back_sigma = background_encoder(Variable(query_img*(1-query_sal)).cuda()) #obtain background features
        
        # Inter-class Hallucination       
        support_fore_features = supp_fore_sigma*torch.rand(supp_fore_sigma.size()).cuda() + supp_fore_mu
        support_back_features = supp_back_sigma*torch.rand(supp_back_sigma.size()).cuda() + supp_back_mu
        support_mix_features = mixture_network(0.2*support_fore_features +  0.8*support_back_features)
        
        
        # No hallucination for query samples
        query_fore_features = query_fore_sigma*torch.rand(query_fore_sigma.size()).cuda() + query_fore_mu
        query_back_features = query_back_sigma*torch.rand(query_back_sigma.size()).cuda() + query_back_mu
        query_mix_features = mixture_network(0.2*query_fore_features + 0.8*query_back_features)
        
        support_img_ = vae_decoder(support_mix_features)
        query_img_ = vae_decoder(query_mix_features)
        
        loss_vae = mse(torch.cat((support_img_,query_img_),0), torch.cat((support_img,query_img),0).cuda())
        
        support_mix_features = support_mix_features.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,-1).mean(1)
        query_mix_features = query_mix_features.view(-1,64*19**2)
        relations = euclidean_metric(query_mix_features,support_mix_features)
        
        '''
        so_support_features = Variable(torch.Tensor(CLASS_NUM*SUPPORT_NUM_PER_CLASS, 1, 64, 64)).cuda()
        so_query_features = Variable(torch.Tensor(QUERY_NUM_PER_CLASS*CLASS_NUM, 1, 64, 64)).cuda()

        
        # second-order features
        for d in range(support_mix_features.size()[0]):
            s = support_mix_features[d,:,:].squeeze(0)
            s = (1.0 / support_mix_features.size()[2]) * s.mm(s.transpose(0,1))
            so_support_features[d,:,:,:] = power_norm(s / s.trace(), SIGMA)
        for d in range(query_mix_features.size()[0]):
            s = query_mix_features[d,:,:].squeeze(0)
            s = (1.0 / query_mix_features.size()[2]) * s.mm(s.transpose(0,1))
            so_query_features[d,:,:,:] = power_norm(s / s.trace(), SIGMA)

        so_support_features = so_support_features.view(CLASS_NUM, -1, 1, 64, 64).mean(1)
        
        # calculate relations with 64x64 second-order features
        support_features_ext = so_support_features.unsqueeze(0).repeat(QUERY_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        query_features_ext = so_query_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        query_features_ext = torch.transpose(query_features_ext,0,1)
        relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,2,64,64)
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
        one_hot_labels = Variable(torch.zeros(QUERY_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, query_labels.view(-1,1), 1)).cuda()
        '''
        
        loss_rn = ce(relations,query_labels.cuda())
        loss = loss_rn + BETA2*loss_vae 
        
        # update network parameters
        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()
        
        if np.mod(episode+1,100)==0:
        	print("episode:",episode+1,"loss: ",loss_rn.item(), "loss vae:",loss_vae.item())
        	
        if np.mod(episode,2500)==0:
            # test
            print("Testing...")
            accuracies = []
            TEST_CLASS_NUM = 5
            for i in range(TEST_EPISODE):
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metaquery_folders,TEST_CLASS_NUM,SUPPORT_NUM_PER_CLASS,15)
                support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SUPPORT_NUM_PER_CLASS,split="train",shuffle=False)
                num_per_class = 3
                query_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True)
                support_img,support_sal,support_labels = support_dataloader.__iter__().next()
                for query_img,query_sal,query_labels in query_dataloader:
                    query_size = query_labels.shape[0]
                    
                    # calculate features      
                    supp_fore_mu, supp_fore_sigma = foreground_encoder(Variable(support_img*support_sal).cuda())
                    supp_back_mu, supp_back_sigma = background_encoder(Variable(support_img*(1-support_sal)).cuda())
                    query_fore_mu, query_fore_sigma = foreground_encoder(Variable(query_img*query_sal).cuda())
                    query_back_mu, query_back_sigma = background_encoder(Variable(query_img*(1-query_sal)).cuda())
                    
                    # No hallucination
                    support_fore_features = supp_fore_sigma*torch.rand(supp_fore_sigma.size()).cuda() + supp_fore_mu
                    support_back_features = supp_back_sigma*torch.rand(supp_back_sigma.size()).cuda() + supp_back_mu
                    support_mix_features = mixture_network(0.2*support_fore_features + 0.8*support_back_features).view(TEST_CLASS_NUM,SUPPORT_NUM_PER_CLASS,64*19*19).mean(1)
                    
                    # No hallucination for query samples
                    query_fore_features = query_fore_sigma*torch.rand(query_fore_sigma.size()).cuda() + query_fore_mu
                    query_back_features = query_back_sigma*torch.rand(query_back_sigma.size()).cuda() + query_back_mu
                    query_mix_features = mixture_network(0.2*query_fore_features + 0.8*query_back_features).view(-1,64*19**2)
                    
                    relations = euclidean_metric(query_mix_features,support_mix_features)
                    
                    # Intra-class Hallucination
                    '''
                    #support_fore_features = support_fore_features.unsqueeze(2).repeat(1,1,SUPPORT_NUM_PER_CLASS,1,1,1)
                    #support_back_features = support_back_features.unsqueeze(1).repeat(1,SUPPORT_NUM_PER_CLASS,1,1,1,1)
                    #support_mix_features =  mixture_network((0.2*support_fore_features + 0.8*support_back_features).view(CLASS_NUM*(SUPPORT_NUM_PER_CLASS**2),64,19,19)).view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,-1,64,19**2).sum(2).sum(1)
                    #query_mix_features = mixture_network(query_fore_features*query_back_features).view(-1,64,19**2)
                    so_support_features = Variable(torch.Tensor(TEST_CLASS_NUM*SUPPORT_NUM_PER_CLASS, 1, 64, 64)).cuda()
                    so_query_features = Variable(torch.Tensor(query_size, 1, 64, 64)).cuda()
                    
        	    	# second-order features
                    for d in range(support_mix_features.size()[0]):
                        s = support_mix_features[d,:,:].squeeze(0)
                        s = (1.0 / support_mix_features.size()[2]) * s.mm(s.transpose(0,1))
                        so_support_features[d,:,:,:] = power_norm(s / s.trace(),SIGMA)
                    for d in range(query_mix_features.size()[0]):
                        s = query_mix_features[d,:,:].squeeze(0)
                        s = (1.0 / query_mix_features.size()[2]) * s.mm(s.transpose(0,1))
                        so_query_features[d,:,:,:] = power_norm(s / s.trace(), SIGMA)
                        
                    # calculate relations with 64x64 second-order features
                    support_features_ext = so_support_features.unsqueeze(0).repeat(query_size,1,1,1,1)
                    query_features_ext = so_query_features.unsqueeze(0).repeat(TEST_CLASS_NUM,1,1,1,1)
                    query_features_ext = torch.transpose(query_features_ext,0,1)
                    relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,2,64,64)
                    relations = relation_network(relation_pairs).view(-1,TEST_CLASS_NUM)
                    '''
                    _,predict_labels = torch.max(relations.data,1)
                    rewards = [1 if predict_labels[j]==query_labels[j].cuda() else 0 for j in range(query_size)]
                    total_rewards += np.sum(rewards)
                    counter += query_size
                    
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)
            test_accuracy,h = mean_confidence_interval(accuracies)
                      
            checkpoint['acc'].append(test_accuracy)
            checkpoint['foreground_encoder'] = foreground_encoder.state_dict()
            checkpoint['background_encoder'] = background_encoder.state_dict()
            checkpoint['vae_decoder'] = vae_decoder.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "checkpoints/" + METHOD + "/checkpoint_up_to_date.pth.tar")
            
            if test_accuracy > best_accuracy:
        	# save networks
                checkpoint_best = {}
                checkpoint_best['foreground_encoder'] = foreground_encoder.state_dict()
                checkpoint_best['background_encoder'] = background_encoder.state_dict()
                checkpoint_best['vae_decoder'] = vae_decoder.state_dict()
                checkpoint_best['optimizer'] = optimizer.state_dict()
                checkpoint_best['acc'] = test_accuracy
                torch.save(checkpoint_best, "checkpoints/" + METHOD + "/checkpoint_best.pth.tar")
                print("save networks for episode:",episode)
                best_accuracy = test_accuracy
                
            print("test accuracy:",test_accuracy,"h:",h)
            print("best accuracy:",best_accuracy)
            
if __name__ == '__main__':
    main()
