import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
从硬截断loss、软化loss发展而来，总体思想就是增加低概率样本（小分类样本）权重，只是调变方式不太一样
L = y[-α(1-p)^γlog(p)] + (1-y)[-(1-α)(p)^γlog(1-p)]
'''

class FocalLoss(nn.Module):
    '''
    alpha代表各类别的权重，如果不指定则为1
    '''
    def __init__(self, gamma=0, alpha=None, size_average=True, class_num=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor(alpha) if alpha else torch.Tensor([1 for _ in range(class_num)])

        self.size_average = size_average

    def forward(self, inputs, target):
        # inputs.dim()>2对这个任务来说不需要考虑，全连接输出的是一维
        if inputs.dim()>2:
            inputs = inputs.view(inputs.size(0),inputs.size(1),-1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1,2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1,inputs.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        # 维度说明：target_shape = (batch_size, 1), input.shape = (batch_size, output_dim)

        # 计算log(softmax(x)), 激活函数二分类可换sigmoid
        logpt = F.log_softmax(inputs)
        # 得到log(yt)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        # 获取yt的概率
        pt = Variable(logpt.data.exp())

        # 确定类别权重参数α
        if self.alpha.type() != inputs.data.type():
            self.alpha = self.alpha.type_as(inputs.data)
        at = self.alpha.gather(0, target.data.view(-1))
        
        loss = -1 * (1-pt)**self.gamma * logpt*Variable(at)
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()

class I_FocalLoss(nn.Module):
    '''
    @citing: 崔子越, et al."结合改进VGGNet和Focal Loss的人脸表情识别." 计算机工程与应用 57.19(2021):171-178
    针对Focal Loss无法处理误标注样本问题，通过样本的置信度与真实标签对其设置阈值判断，
    对误标注样本进行筛选，改变其置信度，从而降低Focal Loss对该类样本关注度，提高模型分类性能。
    '''
    def __init__(self, gamma=0, alpha=None, size_average=True, class_num=2, c=0.5):
        super(I_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor(alpha) if alpha else torch.Tensor([1 for _ in range(class_num)])
        self.c = c
        self.size_average = size_average

    def forward(self, inputs, target):
        target = target.view(-1,1)

        # 计算log(softmax(x)), 激活函数二分类可换sigmoid
        logpt = F.log_softmax(inputs)
        # 得到log(yt)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        # 获取yt的概率
        pt = Variable(logpt.data.exp())

        # 确定类别权重参数α
        if self.alpha.type() != inputs.data.type():
            self.alpha = self.alpha.type_as(inputs.data)
        at = self.alpha.gather(0, target.data.view(-1))
        
        loss = -1 * (1-pt)**self.gamma * logpt*Variable(at)

        # 筛选出置信度高且判断错误的样本，为其赋予极小值
        pt = F.softmax(inputs)
        for i in range(len(target)):
            ptop_label = torch.argmax(pt[i])
            ptop = torch.max(pt[i])
            if ptop > self.c and ptop_label != target[i].data:
                loss[i] = 1e-5

        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()

class Heuristic_FocalLoss(nn.Module):
    '''
    @citing: 傅博文.基于CNN的图像情感分析研究.2020.杭州电子科技大学
    启发式focal loss，α与γ自动寻优
    '''
    def __init__(self, gamma, alpha, size_average=True):
        super(Heuristic_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        # Heuristic_FocalLoss根据epoch修改γ值，在epoch对gamma参数进行更新
        # 更新公式 heuristic_focalLoss.gamma *= .2 * (epoch // num_epochs) + 0.4
        target = target.view(-1,1)

        # 计算log(softmax(x)), 激活函数二分类可换sigmoid
        logpt = F.log_softmax(inputs)
        # 得到log(yt)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        # 获取yt的概率
        pt = Variable(logpt.data.exp())

        # 确定类别权重参数α
        if self.alpha.type() != inputs.data.type():
            self.alpha = self.alpha.type_as(inputs.data)
        at = self.alpha.gather(0, target.data.view(-1))
        
        loss = -1 * (1-pt)**self.gamma * logpt*Variable(at)


        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()