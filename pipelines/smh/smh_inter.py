import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision
import numpy as np
import task_generator as tg
import os
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
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--support_num_per_class",type = int, default = 1)
parser.add_argument("-q","--query_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-sigma","--sigma", type = float, default = 150)
parser.add_argument("-beta","--beta", type = float, default = 0.2)
parser.add_argument("-alpha","--alpha", type = float, default = 1)
args = parser.parse_args()

# Hyper Parameters
METHOD = "smh_inter_beta" + str(args.beta)
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SUPPORT_NUM_PER_CLASS = args.support_num_per_class
BATCH_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
SIGMA = args.sigma

def similarity_func(x,c,s):
	score = torch.Tensor(c,s,c*s).cuda(GPU)
	for c_ in range(c):
		for s_ in range(s):
			metric = (torch.abs(x[c_,s_,c_*s+s_,:,:,:] - x[c_,s_,:,:,:,:])**2)
			metric = metric.mean(3).mean(2).mean(1)
			score[c_,s_,:] = 2*torch.exp(-args.alpha*metric)/(1 + torch.exp(-args.alpha*metric))
	return score
	
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
    foreground_encoder = models.FeatureEncoder().apply(weights_init).cuda(GPU)
    background_encoder = models.FeatureEncoder().apply(weights_init).cuda(GPU)
    mixture_network = models.MixtureNetwork(64).apply(weights_init).cuda(GPU)
    relation_network = models.SimilarityNetwork(FEATURE_DIM,RELATION_DIM).apply(weights_init).cuda(GPU)

    rd_foreground_encoder = models.FeatureEncoder().apply(weights_init).cuda(GPU)
    rd_background_encoder = models.FeatureEncoder().apply(weights_init).cuda(GPU)
    rd_mixture_network = models.MixtureNetwork(64).apply(weights_init).cuda(GPU)        

    optimizer = torch.optim.Adam([{'params': foreground_encoder.parameters()}, 
                                 {'params': background_encoder.parameters()},
                                 {'params': mixture_network.parameters()},
                                 {'params': relation_network.parameters()}], lr=LEARNING_RATE)

    optimizer_scheduler = StepLR(optimizer,step_size=100000,gamma=0.5)
	
	# Loading models
    if os.path.exists("checkpoints/" + METHOD + "/chechpoint_" + str(SHOT) + "shot_up_to_date.pth.tar"):
    	checkpoint = torch.load("checkpoints/" + METHOD + "/chechpoint_" + str(SHOT) + "shot_up_to_date.pth.tar")
    	foreground_encoder.load_state_dict(checkpoint['foreground_encoder'])
    	background_encoder.load_state_dict(checkpoint['background_encoder'])
    	mixture_network.load_state_dict(checkpoint['mixture_network'])
    	relation_network.load_state_dict(checkpoint['relation_network'])
    	optimizer.load_state_dict(checkpoint['optimizer'])
    	print("load modules successfully!")
        
    # Loading vanilla models    
    if os.path.exists("checkpoints/" + METHOD + "/chechpoint_" + str(SHOT) + "shot_rd.pth.tar"):
    	checkpoint = torch.load("checkpoints/" + METHOD + "/chechpoint_" + str(SHOT) + "shot_rd.pth.tar")
    	rd_foreground_encoder.load_state_dict(checkpoint['foreground_encoder'])
    	rd_background_encoder.load_state_dict(checkpoint['background_encoder'])
    	rd_mixture_network.load_state_dict(checkpoint['mixture_network'])
    	print("load modules successfully!") 
    	      
    if os.path.exists('checkpoints') == False:
        os.system('mkdir checkpoints')
        
    if os.path.exists('checkpoints/' + METHOD) == False:
        os.system('mkdir checkpoints/' + METHOD)

    print("Training...")

    best_accuracy = 0.0
    start = time.time()            
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
        s_fore_feat = foreground_encoder(Variable(support_img*support_sal).cuda(GPU)) #obtain foreground features
        s_back_feat = background_encoder(Variable(support_img*(1-support_sal)).cuda(GPU)) #obtain background features
        q_fore_feat = foreground_encoder(Variable(query_img*query_sal).cuda(GPU)) #obtain foreground features
        q_back_feat = background_encoder(Variable(query_img*(1-query_sal)).cuda(GPU)) #obtain background features
        
        # Representation Distillation (RD), teacher network
        s_fore_feat_ = rd_foreground_encoder(Variable(support_img*support_sal).cuda(GPU)) #vanilla foreground representations
        s_back_feat_ = rd_background_encoder(Variable(support_img*(1-support_sal)).cuda(GPU)) #vanilla background representation
        s_mix_feat_ = rd_mixture_network(args.gamma*s_fore_feat_+(1-args.gamma)*s_back_feat_)     
        s_mix_feat__ = mixture_network(args.gamma*s_fore_feat+(1-args.gamma)*s_back_feat) 
        RD = args.beta*mse(s_mix_feat__, Variable(s_mix_feat_, requires_grad=False))
        
        # Inter-class Hallucination
        s_fore_feat = s_fore_feat.view(CLASS_NUM, SUPPORT_NUM_PER_CLASS,64,19,19)
        s_back_feat = s_back_feat.view(CLASS_NUM, SUPPORT_NUM_PER_CLASS,64,19,19)
        s_fore_feat = s_fore_feat.unsqueeze(2).repeat(1,1,CLASS_NUM*SUPPORT_NUM_PER_CLASS,1,1,1)
        s_back_feat = s_back_feat.view(1,1,CLASS_NUM*SUPPORT_NUM_PER_CLASS,64,19,19).repeat(CLASS_NUM,SUPPORT_NUM_PER_CLASS,1,1,1,1)
        sim_measure = similarity_func(s_back_feat, CLASS_NUM, SUPPORT_NUM_PER_CLASS).view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,-1,1,1)
        
        s_mix_feat = mixture_network((args.gamma*s_fore_feat + (1-args.gamma)*s_back_feat).view((CLASS_NUM**2)*(SUPPORT_NUM_PER_CLASS**2),128,19,19)).view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,-1,64,19**2)
        s_mix_feat = (s_mix_feat*sim_measure).view(-1,64,19**2)
        
        # No hallucination for query samples
        q_mix_feat = mixture_network(args.gamma*q_fore_feat + (1-args.gamma)*q_back_feat).view(-1,64,19**2)
        so_s_feat = Variable(torch.Tensor(s_mix_feat.size()[0], 1, 64, 64)).cuda(GPU)
        so_q_feat = Variable(torch.Tensor(QUERY_NUM_PER_CLASS*CLASS_NUM, 1, 64, 64)).cuda(GPU)

        # second-order features
        for d in range(s_mix_feat.size()[0]):
            s = s_mix_feat[d,:,:].squeeze(0)
            s = (1.0 / s_mix_feat.size()[2]) * s.mm(s.transpose(0,1))
            so_s_feat[d,:,:,:] = power_norm(s / s.trace(), SIGMA)
        for d in range(q_mix_feat.size()[0]):
            s = q_mix_feat[d,:,:].squeeze(0)
            s = (1.0 / q_mix_feat.size()[2]) * s.mm(s.transpose(0,1))
            so_q_feat[d,:,:,:] = power_norm(s / s.trace(), SIGMA)

        so_s_feat = so_s_feat.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,-1,1,64,64).mean(2).mean(1)
        # calculate relations with 64x64 second-order features
        s_feat_ = so_s_feat.unsqueeze(0).repeat(QUERY_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        q_feat_ = so_q_feat.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        q_feat_ = torch.transpose(q_feat_,0,1)
        relation_pairs = torch.cat((s_feat_,q_feat_),2).view(-1,2,64,64)
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

        loss = ce(relations, query_labels.cuda()) + RD
        
        # update network parameters
        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()
        
        if np.mod(episode+1,100)==0:
        	print("episode:",episode+1,"loss",loss.item())
        	
        if np.mod(episode,2500)==0:
            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metaquery_folders,CLASS_NUM,SUPPORT_NUM_PER_CLASS,15)
                support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SUPPORT_NUM_PER_CLASS,split="train",shuffle=False)
                num_per_class = 3
                query_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True)
                support_img,support_sal,support_labels = support_dataloader.__iter__().next()
                for query_img,query_sal,query_labels in query_dataloader:
                    query_size = query_labels.shape[0]
                    
        	    	# calculate foreground and background features
                    s_fore_feat= foreground_encoder(Variable(support_img*support_sal).cuda(GPU)).view(CLASS_NUM, SUPPORT_NUM_PER_CLASS,64,19,19)
                    s_back_feat= background_encoder(Variable(support_img*(1-support_sal)).cuda(GPU)).view(CLASS_NUM, SUPPORT_NUM_PER_CLASS,64,19,19)
                    q_fore_feat = foreground_encoder(Variable(query_img*query_sal).cuda(GPU))
                    q_back_feat = background_encoder(Variable(query_img*(1-query_sal)).cuda(GPU))
                    
        	    	# Intra-class Hallucination
                    s_fore_feat = s_fore_feat.view(CLASS_NUM, SUPPORT_NUM_PER_CLASS,64,19,19)
                    s_back_feat = s_back_feat.view(CLASS_NUM, SUPPORT_NUM_PER_CLASS,64,19,19)
                    s_fore_feat = s_fore_feat.unsqueeze(2).repeat(1,1,CLASS_NUM*SUPPORT_NUM_PER_CLASS,1,1,1)
                    s_back_feat = s_back_feat.view(1,1,CLASS_NUM*SUPPORT_NUM_PER_CLASS,64,19,19).repeat(CLASS_NUM,SUPPORT_NUM_PER_CLASS,1,1,1,1)
                    sim_measure = similarity_func(s_back_feat, CLASS_NUM, SUPPORT_NUM_PER_CLASS).view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,-1,1,1)
                    
                    s_mix_feat = mixture_network((args.gamma*s_fore_feat + (1-args.gamma)*s_back_feat).view((CLASS_NUM**2)*(SUPPORT_NUM_PER_CLASS**2),128,19,19)).view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,-1,64,19**2)
                    s_mix_feat = (s_mix_feat*sim_measure).view(-1,64,19**2)
                    
                    q_mix_feat = mixture_network(args.gamma*q_fore_feat + (1-args.gamma)*q_back_feat).view(-1,64,19**2)
                    so_s_feat = Variable(torch.Tensor(s_mix_feat.size()[0], 1, 64, 64)).cuda(GPU)
                    so_q_feat = Variable(torch.Tensor(query_size, 1, 64, 64)).cuda(GPU)
                    
        	    	# second-order features
                    for d in range(s_mix_feat.size()[0]):
                        s = s_mix_feat[d,:,:].squeeze(0)
                        s = (1.0 / s_mix_feat.size()[2]) * s.mm(s.transpose(0,1))
                        so_s_feat[d,:,:,:] = power_norm(s / s.trace(),SIGMA)
                    for d in range(q_mix_feat.size()[0]):
                        s = q_mix_feat[d,:,:].squeeze(0)
                        s = (1.0 / q_mix_feat.size()[2]) * s.mm(s.transpose(0,1))
                        so_q_feat[d,:,:,:] = power_norm(s / s.trace(), SIGMA)
                        
                    so_s_feat = so_s_feat.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,-1,1,64,64).mean(2).mean(1)
                    # calculate relations with 64x64 second-order features
                    s_feat_ = so_s_feat.unsqueeze(0).repeat(query_size,1,1,1,1)
                    q_feat_ = so_q_feat.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
                    q_feat_ = torch.transpose(q_feat_,0,1)
                    relation_pairs = torch.cat((s_feat_,q_feat_),2).view(-1,2,64,64)
                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
                    _,predict_labels = torch.max(relations.data,1)
                    rewards = [1 if predict_labels[j]==query_labels[j].cuda(GPU) else 0 for j in range(query_size)]
                    total_rewards += np.sum(rewards)
                    counter += query_size
                    
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)
            test_accuracy,h = mean_confidence_interval(accuracies)
            
            checkpoint['acc'].append(test_accuracy)
            checkpoint['foreground_encoder'] = foreground_encoder.state_dict()
            checkpoint['background_encoder'] = background_encoder.state_dict()
            checkpoint['mixture_network'] = mixture_network.state_dict()
            checkpoint['relation_network'] = relation_network.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "checkpoints/" + METHOD + "/chechpoint_" + str(SHOT) + "shot_up_to_date.pth.tar")
            
            if test_accuracy > best_accuracy:
        	# save networks
                checkpoint_best = {}
                checkpoint_best['foreground_encoder'] = foreground_encoder.state_dict()
                checkpoint_best['background_encoder'] = background_encoder.state_dict()
                checkpoint_best['mixture_network'] = mixture_network.state_dict()
                checkpoint_best['relation_network'] = relation_network.state_dict()
                checkpoint_best['optimizer'] = optimizer.state_dict()
                checkpoint_best['acc'] = test_accuracy
                torch.save(checkpoint_best, "checkpoints/" + METHOD + "/chechpoint_" + str(SHOT) + "shot_best.pth.tar")
                print("save networks for episode:",episode)
                best_accuracy = test_accuracy
                
            print("test accuracy:",test_accuracy,"h:",h)
            print("best accuracy:",best_accuracy)
            
if __name__ == '__main__':
    main()
