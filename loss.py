import torch.nn.functional as F
import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, z1, z2, target):
        distances = (z2 - z1).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances + (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean()

#old loss
class SpectralNetLoss(nn.Module):
    
    def __init__(self):
        super(SpectralNetLoss, self).__init__()
        
    def forward(self, Y, W, X):
        # W是权重，其shape为[1024, 10]，Y通过模型得到的特征向量，其shape为[1024, 10]
        Yd = torch.cdist(Y, Y, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')**2
        #add by lwb torch.cdist 通过矩阵乘法计算 欧式距离 yd=[1024, 1024]
        # return torch.sum(W*Yd)/(W.shape[0])+torch.norm(Y@Y.transpose(1,0), p=1 )/(W.shape[0]) # 添加的一范数torch.norm(Y*Y.t, p=1 ) add by lwb
        return torch.sum(W*Yd)/(W.shape[0])

#new loss add by lwb transformer
# class SpectralNetLoss(nn.Module):
    
#     def __init__(self):
#         super(SpectralNetLoss, self).__init__()
        
#     def forward(self, Y, W, X):
#         Yd = torch.cdist(Y, Y, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')**2
#         return torch.sum(W*Yd)/(W.shape[0])

        # tr(FLF) 部分
        # W是权重，其shape为[1024, 10]，Y通过模型得到的特征向量，其shape为[1024, 10] 
        #new loss  ||X -FF.tX|| 并且F>0 部分 # 通过计算样本 和 近似特征矩阵损失
        # Y = 1.0/2*(Y + torch.abs(Y))
        # sample_loss = X- torch.mm(torch.mm(Y ,Y.T),X)
        # sample_loss = 1./(sample_loss.shape[0]**2) * sample_loss 
        # return torch.norm(sample_loss, p=2)+torch.sum(W*Yd)/(W.shape[0])
        # Y = 1.0/2*(Y + torch.abs(Y))
        # sample_loss = X- torch.mm(torch.mm(Y ,Y.T),X)
        # sample_loss = 1./(sample_loss.shape[0]**2) * sample_loss 
        # 将样本和 近似的做 做模运算
        #Y是通过近似得到的结 m*k
        # 之前的loss train_loss = 46.758804 都是在 上万经过归一化处理 loss 降低了
        # 推测有问题， 如果在这列设置F 大于0 会不会将所有的F值都改变了;  --问题正确， 解决
        # 如果将小于0 的值修改了之后会不会不正交两个问题。问题正确，的确遇到了。
        # 对的确不能够直接改F==Y 。 因为这个取值一旦修改了之后，那么这个模型的参数就会无法正常的去更新梯度了。
        # F[F<0] = 0
        # sample_loss = X- torch.mm(torch.mm(F ,F.T),X)# 通过计算样本 和 近似特征矩阵损失
        # sample_loss = 1./(sample_loss.shape[0]**2) * sample_loss # 归一化处理？ 
        #origin
        # return torch.norm(X-Y, p=2)+torch.sum(W*Yd)/(W.shape[0])