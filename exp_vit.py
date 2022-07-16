import torch
import torchvision 
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import keras

import numpy as np

from metrics import acc
from sklearn.metrics import normalized_mutual_info_score

from loss import SpectralNetLoss


from data import get_embedding
from data import siamesedataloader
from data import to_graph
from vit_model.load_data import load_data
from trainbyVit import  train_SpectralNet_vit
from sklearn.metrics import normalized_mutual_info_score

from sklearn.model_selection import train_test_split

from spectral_methods import SpectralClustering
from sklearn.cluster import KMeans

'''
pip install keras==2.3
pip install tensorflow==1.15
pip install annoy
'''
import exportExcel

def run_experiments(data=None, n_clusters=None, embedding=True, train_own_AE = False, siam=True, batch_size= 1024 , n_neighbors=2, aprox=False):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load data
  trainset, testset  = load_data(batch_size=batch_size)
  train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,)
  ortho_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,)

  val_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
  test_dataloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True)

  #train model
  print("train TranSpectral")
  sheet_num ="010"
  model_spectralN = train_SpectralNet_vit(
    output_size=n_clusters, 
    batch_size=batch_size, 
    file='checkpoint/training'+sheet_num+'.pkl', 
    n_neighbors=n_neighbors, 
    aprox=aprox, 
    train_dataloader=train_dataloader, 
    val_dataloader=val_dataloader, 
    ortho_dataloader = ortho_dataloader,
    epochs=2,
    sheet_name="training"+sheet_num,
    lr=1e-3,
    depth=6,
    verbose=True)
  # reload model
  # model_spectralN=torch.load('checkpoint/spectralNet_transformer_epoch2.pkl')
  datalist = []

  # test model
  # TODO:验证数据集的size 不一样害怕出错，现在思考如何将这个数据集全部弄成tansform之后的格式
  #SpectralNet
  with torch.no_grad():
    X = []
    Y = []
    for x, target in test_dataloader:
        X= model_spectralN(x.to(device)).detach().to('cpu').numpy()
        Y= target.detach().to('cpu').numpy()
    print("SpectralNetloss_transformer_keams")
    # spectralNet _kmeansi
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    NMI_rs = normalized_mutual_info_score(Y,kmeans.labels_)
    acc_rs = acc(Y, kmeans.labels_)
    print(f"NMI = {NMI_rs}")
    print(f"ACC = {acc_rs}")
    datalist.append([ 'ours_kmeans', round(NMI_rs, 4), round(acc_rs, 4)])
  
    print("SpectralNetloss_transformer_argmax")
    max_rs = X.argmax(1)
    NMI_rs=normalized_mutual_info_score(Y,max_rs)
    acc_rs = acc(Y, max_rs)
    print(f"NMI = {NMI_rs}")
    print(f"ACC = {acc_rs}")
    datalist.append([ 'ours_argmax', round(NMI_rs,4), round(acc_rs,4)])
  

    # exportExcel.export_excel(sheet_name="st_orthdata_v1", col=('clusteringMethod', "NMI", "ACC"),datalist=datalist)
    #2022-7-11
    exportExcel.export_excel(sheet_name="result"+sheet_num, col=('clusteringMethod', "NMI", "ACC"),datalist=datalist)
  print('111')
  #return model_spectralN

if __name__ == "__main__":
     run_experiments(data="mnist", n_clusters=10, embedding=True, train_own_AE = False, siam=False, aprox=False)
  
