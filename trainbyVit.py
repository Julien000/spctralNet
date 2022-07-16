import torch
from torch import nn
import torch.optim as optim

from loss import ContrastiveLoss
from loss_vit import SpectralNetLoss

from nets import SiameseNet
from nets import AE
from nets import SpectralNet

from data import to_graph
from sklearn.model_selection import train_test_split
import exportExcel
from vit_model.load_data import load_data

from vit_model.model import SpectralTransformer


def train_SpectralNet_vit(
  output_size,
  batch_size, 
  n_neighbors,
  train_dataloader,
  ortho_dataloader,
  val_dataloader, 
  aprox = False, 
  model_siam=None, 
  file = None, 
  epochs = 3, 
  sheet_name ="training",
  lr = 1e-5,
  depth= 12,
  verbose=False):

  # configuration
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # 加载transformer模型，设置优化策略 
  model = SpectralTransformer(output_size=output_size, device=device, depth=depth)

  optimizer = optim.Adam(model.parameters(), lr=lr) 
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  # TODO: 开始修改loss ，基于transformer 的 交叉熵损失函数，
  # criterion = SpectralNetLoss()
  criterion = nn.CrossEntropyLoss()


  datalist =[]
  for epoch in range(epochs):
    loss_t = 0
    for zipdata in zip(train_dataloader,ortho_dataloader):
      x1, x2 = zipdata[0][0], zipdata[1][0]
     
      #1. ortostep  正交分解部分
      model.eval()
      _ = model(x2.to(device),ortho_step=True)

      #2. gradstep
      model.train()
      optimizer.zero_grad()#提前将梯度清空，防止累积梯度 add by lwb 好的习惯在模型进行下次计算前清空梯度
      x1 = x1.to(device)
      Y = model(x1)

      #3.  构造利用knn 构造W 矩阵，
      W = to_graph(x1.detach().to("cpu").numpy().reshape(x1.shape[0],-1),"mean",None,n_neighbors,'k-hNNG',aprox).todense()
      W = torch.from_numpy(W).to(device)

      # 4. 计算loss add by lwb
      # train_loss = criterion(Y,W, x1) # 这个东西的功能就是求解正则项---torch.cdist---求解正则项p-norm
      Yd = torch.cdist(Y, Y, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')**2
      train_loss = criterion(Yd,W) 
      print(train_loss)  
      train_loss.backward()
      optimizer.step()

      loss_t += train_loss.item()

    loss_t = loss_t / len(train_dataloader)
    #valid 验证模型
    model.eval()
    loss_v = 0
    with torch.no_grad():
      for x, target in val_dataloader:
        x = x.to(device)
        Y = model(x)
        W = to_graph(x.detach().to("cpu").numpy().reshape(x.shape[0],-1),"mean",None,n_neighbors,'k-hNNG',aprox).todense()
        W = torch.from_numpy(W).to(device)
              
        # val_loss = criterion(Y,W)
        Yd = torch.cdist(Y, Y, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')**2
        val_loss = criterion(Yd,W) # add by lwb
              
        loss_v += val_loss.item()
      
      loss_v = loss_v / len(val_dataloader)
      scheduler.step(loss_v)
          
      act_lr = optimizer.param_groups[0]['lr']
      if verbose:
        print("epoch : {}/{}, learning_rate {}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, epochs,act_lr, loss_t, loss_v))
        datalist.append([epoch+1, act_lr, round(loss_t,4), round(loss_v, 4)] )
      if act_lr <= 1e-7:
        break
  if file!= None:
    torch.save(model, file)
  
  #TODOnew: 
  exportExcel.export_excel(sheet_name=sheet_name, col=('epoch',' learning_rate', 'train_loss', 'val_loss'),datalist=datalist)
  # exportExcel.export_excel(sheet_name="train_depth4_lr1e5_spectralloss_bs256_epoch100", col=('epoch',' learning_rate', 'train_loss', 'val_loss'),datalist=datalist)
  return model
  # fmo 代表F>0
# 总结一下，当前的loss 函数似乎还是没有其效果，最开始的loss 几乎就是0.0几开始，
# 然后结果当然是不太理想的，然后不知道应该如何处理，这个东西，
# 由于采用的是聚类算法，这个是对整个数据集进行模拟， 现在是要根据spctralNet 的权重矩阵进行模拟。
