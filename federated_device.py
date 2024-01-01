import torch
import math
from torch.utils.data import DataLoader
import numpy as np
from simclr import SimCLR
from simclr.modules import NT_Xent
from torchvision import datasets, transforms
import torchvision 
import pandas as pd
from simclr.modules.transformations import TransformsSimCLR
from sklearn.cluster import AgglomerativeClustering
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import torch.nn as nn
from sklearn.datasets import make_blobs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''************************服务器客户端共有操作********************************'''
def eval_op(model,loader):
    model.eval()
    samples, correct = 0.0, 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            _, predicted = torch.max(y_hat.data, 1)
            samples += y.shape[0]
            correct += (predicted == y).sum().item()
    ## 返回准确率
    return correct/samples


'''************************客户端操作********************************'''
 # 计算差并将结果存储到目标
def subtract_(W, W_old):
    dW = {key : torch.zeros_like(value) for key, value in W.items()}
    for name in W:
        dW[name].data = W[name].data.clone()-W_old[name].data.clone()
    return dW

# 添加拉普拉斯噪声
def add_laplace_noise(data, sensitivity, epsilon):
    beta = sensitivity / epsilon
    noise = np.random.laplace(0, beta, len(data))
    return data + noise
'''***************************客户端******************************************'''
class Client():
    def __init__(self, model, data, client_id, client_model, batch_size=128,train_frac=0.8):
        super().__init__()  
        self.Client_net = model.to(device)
        self.Client_data = data
        self.Client_id = client_id
        self.Client_model = client_model
        self.W = {key : torch.zeros_like(value) for key, value in self.Client_net.named_parameters()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.Client_net.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.Client_net.named_parameters()}
        '''**********************************数据加载*******************************************'''
        # 80%的数据为训练集，20%为测试集
        n_train = int(len(data)*train_frac)
        n_eval = len(data) - n_train 
        data_train, data_eval = torch.utils.data.random_split(self.Client_data, [n_train, n_eval])
        #获取数据集的大小
        self.data_size =len(data_train)

        self.Client_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.Client_eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=True) 
        '''**********************************数据分布*******************************************'''
        '''
        #获得客户端数据分布数量
        self.class_counts = np.zeros(10)
        #获取类别数量
        for _, y in self.Client_train_loader:
            for i in range(len(y)):
                self.class_counts[y[i]]+=1
        
        #获得客户端数据分布比例
        self.class_rate = np.zeros(10)
        for i in range(len(self.class_counts)):
            self.class_rate[i] = self.class_counts[i]/self.data_size

        # 差分隐私
        sensitivity = 20  # 数据的敏感度，这里假设数据范围为[0, 20]
        epsilon = 0.5  # 隐私预算，控制噪声的强度

        # 对客户端数据添加差分隐私噪声
        self.private_class = add_laplace_noise(self.class_counts, sensitivity, epsilon)        
        '''
    # 训练客户端模型并且计算权重更新
    def compute_weight_update(self, epochs):
        self.W_old = {key : value.clone() for key, value in self.W.items()}#保存旧模型参数
        '''****************训练***************************'''
        if self.Client_model=='MobileNetV2':
            self.optimizer = torch.optim.SGD(self.Client_net.parameters(),lr=0.01, momentum=0.9,weight_decay=1e-6)         
        else:
            
            self.optimizer =torch.optim.SGD(self.Client_net.parameters(), lr=0.01,momentum=0.9,weight_decay=1e-6)
        self.Client_net.train()  
        loss_epoch =0.0
        for _ in range(epochs):
            for x, y in self.Client_train_loader: 
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(self.Client_net(x), y)
                loss.backward()
                self.optimizer.step()  
                loss_epoch += loss.item()
        '''****************训练***************************'''
        self.W = {key : value for key, value in self.Client_net.named_parameters()} #获得新模型参数
        self.dW = subtract_(self.W, self.W_old) #获得参数更新
        
        train_acc = eval_op(model=self.Client_net,loader=self.Client_train_loader)
        test_acc = eval_op(model=self.Client_net,loader=self.Client_eval_loader)
        return loss_epoch/epochs,train_acc,test_acc
        
    
    #定义了一个与服务器同步的方法，该方法将服务器端的权重复制到客户端的权重上。 
    def synchronize_with_server(self, server):
        model = server.mapping[self.Client_id]
        self.W = {key : value.clone() for key, value in model.items()}
        
        for name in self.W:
            self.Client_net.state_dict()[name].copy_(self.W[name].data)    

    def eval(self):
        acc = eval_op(model=self.Client_net,loader=self.Client_eval_loader)
        return acc

    def train_eval(self):
        acc = eval_op(model=self.Client_net,loader=self.Client_train_loader)
        return acc
    '''
    def get_classify(self,server):
        class_rate = self.class_rate
        #计算d
        if(self.Client_model=='MobileNetV2'):
            for i in range(len(class_rate)):
                class_rate[i] = (class_rate[i] - server.global_MobileNetV2_classify[i])**2
            self.d = math.sqrt(class_rate.sum())
        else:
            for i in range(len(class_rate)):
                class_rate[i] = (class_rate[i] - server.global_ResNet8_classify[i])**2
            self.d = math.sqrt(class_rate.sum())
    '''

import random
'''***************************服务器******************************************'''
def flatten(source):
    # # 将源的数据展平并返回
    return torch.cat([value.flatten() for value in source.values()])

def self_train_op(model,optimizer, criterion, loader, model_name):
    print("正在预训练！")
    for _ in range(1000):
        loss_epoch =0.0
        for _, ((x_i, x_j), _) in enumerate(loader):
            optimizer.zero_grad()
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            # positive pair, with encoding
            _, _, z_i, z_j = model(x_i, x_j)
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        print('loss:',loss_epoch )
    print("预训练完成！")

    save_path = 'Pretrained/temp/' + model_name + '_4.pt' 
    torch.save(model.state_dict(), save_path)


# 重写转换函数，增加批量归一化函数。
class CustomTransformsSimCLR(TransformsSimCLR):
    def __init__(self, size):
        super().__init__(size)  # 调用父类的构造函数
        self.additional_transform = transforms.Compose([
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __call__(self, x):
        x1, x2 = super().__call__(x)  # 调用父类的__call__方法
        x1 = self.additional_transform(x1)
        x2 = self.additional_transform(x2)
     
        return x1, x2
    
def pairwise_angles(sources):
    dict =[]
    for _, source in enumerate(sources):
        s = flatten(source)
        dict.append(s)
    Mix = torch.cat(dict,dim=0)
    Mix = Mix.reshape(10,-1)
    return Mix.cpu().numpy()
    

class Server():
    def __init__(self,MobileNetV2,ResNet8):
        super().__init__()
        self.Server_MobileNetV2 = MobileNetV2.to(device)
        self.Server_ResNet8 = ResNet8.to(device)
        self.model_cache = []
        '''
        #如果不使用预训练模型，则使用pytorch默认的初始化。
        self.ResNet8_W = {key : value for key, value in self.Server_ResNet8.named_parameters()}
        self.MobileNetV2_W = {key : value for key, value in self.Server_MobileNetV2.named_parameters()}
        '''
        #建立客户端和模型的映射
        self.mapping = {}
 
    def Pretraining(self,pretrained):
        # 如果已有模型就用训练好的，否则重新训练。
        if(pretrained==1):
            print('正在加载MobileNet模型')
            load_path_MobileNet = 'Pretrained/MobileNet/MobileNetV2_3.pt'
            MobileNetV2 = torch.load(load_path_MobileNet)
            model_dict = self.Server_MobileNetV2.state_dict()
            
            # 从表格文件中读取参数名称的映射关系
            mapping_file_path = 'Pretrained/MobileNet/mapping.csv'
            mapping_df = pd.read_csv(mapping_file_path,delimiter=',')
            
            # 创建参数名称的映射关系
            mapping_MobileNetV2 = {}
            for _, row in mapping_df.iterrows():
                mapping_MobileNetV2[row['pretrained']] = row['model']
           
            
            # 根据映射关系更新预训练模型参数的名称
            mapped_pretrained_dict = {}
            for k, v in MobileNetV2.items():
                if k in mapping_MobileNetV2:
                    mapped_pretrained_dict[mapping_MobileNetV2[k]] = v
            model_dict.update(mapped_pretrained_dict)
            self.Server_MobileNetV2.load_state_dict(model_dict)

            print('正在加载ResNet8模型')
            load_path_ResNet = 'Pretrained/ResNet/ResNet8_3.pt'
            ResNet8 = torch.load(load_path_ResNet)
            
            model_dict = self.Server_ResNet8.state_dict()
            
            # 从表格文件中读取参数名称的映射关系
            mapping_file_path = 'Pretrained/ResNet/mapping.csv'
            mapping_df = pd.read_csv(mapping_file_path,delimiter='\t')

            # 创建参数名称的映射关系
            mapping_ResNet8 = {}
            for _, row in mapping_df.iterrows():
                mapping_ResNet8[row['pretrained']] = row['model']

            # 根据映射关系更新预训练模型参数的名称
            mapped_pretrained_dict = {}
            for k, v in ResNet8.items():
                if k in mapping_ResNet8:
                    mapped_pretrained_dict[mapping_ResNet8[k]] = v
            model_dict.update(mapped_pretrained_dict)
            self.Server_ResNet8.load_state_dict(model_dict)

            self.MobileNetV2_W = {key : value for key, value in self.Server_MobileNetV2.named_parameters()}
            self.ResNet8_W = {key : value for key, value in self.Server_ResNet8.named_parameters()}
           
        else:
            batch_size=128
            # 下载和加载训练集
            trainset = torchvision.datasets.CIFAR100(root='cifar100', train=True,
                                                    download=True, transform=CustomTransformsSimCLR(size=32))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=True, num_workers=4,drop_last=True)
            #损失函数
            criterion = NT_Xent(batch_size=batch_size,temperature=0.5,world_size=1)

            #无监督训练MLP
            model_ResNet8 = SimCLR(encoder=self.Server_ResNet8, projection_dim=100,n_features=64).to(device)
            optimizer_ResNet8 = torch.optim.Adam(model_ResNet8.parameters(), lr=0.001, weight_decay=1e-6)
            self_train_op(model_ResNet8, optimizer_ResNet8, criterion, loader,model_name='ResNet8')

            #无监督训练CNN
            model_MobileNetV2 = SimCLR(encoder=self.Server_MobileNetV2, projection_dim=100,n_features=1024).to(device)
            optimizer_MobileNetV2 = torch.optim.Adam(model_MobileNetV2.parameters(), lr=0.001, weight_decay=1e-6)
            self_train_op(model_MobileNetV2, optimizer_MobileNetV2, criterion, loader,model_name='MobileNetV2')
            
            print('所有模型预训练完成')

    #定义了一个选择客户端的方法，该方法从客户端列表中随机选择一部分客户端，并返回选择的客户端列表。
    def select_clients(self, clients, frac):
        #刚开始时，第一轮训练。
        if (self.mapping=={}):
            list_MobileNetV2 = [client.Client_id for client in clients if client.Client_model=='MobileNetV2']
            list_ResNet8 = [client.Client_id for client in clients if client.Client_model=='ResNet8']
            for i in range(len(list_MobileNetV2)):
                self.mapping[list_MobileNetV2[i]]=self.MobileNetV2_W
            for i in range(len(list_ResNet8)):
                self.mapping[list_ResNet8[i]]=self.ResNet8_W

            '''
            #获取客户端数据分布，计算全局数据分布。
            MobileNetV2_classify = [client.private_class for client in clients if client.Client_model=='MobileNetV2']
            row_sum = np.sum(MobileNetV2_classify, axis=0)
            self.global_MobileNetV2_classify = [ i/sum(row_sum)  for i in row_sum]

            ResNet8_classify = [client.private_class for client in clients if client.Client_model=='ResNet8']
            row_sum = np.sum(ResNet8_classify, axis=0)
            self.global_ResNet8_classify = [ i/sum(row_sum)  for i in row_sum]
            print('global_MobileNetV2_classify: ',self.global_MobileNetV2_classify)
            print('global_ResNet8_classify: ',self.global_ResNet8_classify)
            '''
                
        else:
                #对于新加入的客户端，尚未考虑。
            pass
      
        num_clients = int(len(clients) * frac)
        return clients[:num_clients]
        #return random.sample(clients, int(len(clients)*frac)) 


    #加权聚合
    def aggregate_clusterwise(self, client_MobileNetV2_clusters,client_ResNet8_clusters):
        
        for cluster in client_MobileNetV2_clusters:

            all_data_size = 0.0
            for client in cluster:
                all_data_size += client.data_size
            data_rate = {client.Client_id : client.data_size/all_data_size for client in cluster}

            model = self.mapping[cluster[0].Client_id]        
            for name in model:
                tmp = torch.sum(torch.stack([client.dW[name].data * data_rate[client.Client_id] for client in cluster]), dim=0).clone()
                model[name].data+=tmp
            for client in cluster:
                self.mapping[client.Client_id]=model
    
        for cluster in client_ResNet8_clusters:

            all_data_size = 0.0
            for client in cluster:
                all_data_size += client.data_size
            data_rate = {client.Client_id : client.data_size/all_data_size for client in cluster}

            model = self.mapping[cluster[0].Client_id]        
            for name in model:
                tmp = torch.sum(torch.stack([client.dW[name].data * data_rate[client.Client_id]  for client in cluster ]), dim=0).clone()
                model[name].data+=tmp
            for client in cluster:
                self.mapping[client.Client_id]=model
    

    #定义了一个计算簇中客户端权重更新的最大范数的方法     
    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    #定义了一个计算簇中客户端权重更新的平均范数的方法。
    def compute_mean_update_norm(self, cluster):
        return torch.norm(torch.mean(torch.stack([flatten(client.dW) for client in cluster]), 
                                     dim=0)).item()
    
    def compute_pairwise_similarities(self, clients):
        return pairwise_angles([client.dW for client in clients])

    # 定义了一个缓存模型的方法，该方法将模型的参数、客户端ID和准确率存储到模型缓存列表中。
    def cache_model(self, idcs, accuracies):
        self.model_cache += [(idcs, [accuracies[i] for i in idcs])]

    '''
    def get_n_d_p(self, clients,a,b):
        sum_MobileNetV2 = 0.0
        sum_ResNet8=0.0

        data_size_MobileNetV2 = 0.0
        data_size_ResNet8 = 0.0

        for client in clients:
            if client.Client_model=='MobileNetV2':
                data_size_MobileNetV2+=client.data_size
            else:
                data_size_ResNet8+=client.data_size

        for client in clients:
            if client.Client_model=='MobileNetV2':
                sum_MobileNetV2+=np.maximum(client.data_size/data_size_MobileNetV2 - a * client.d + b, 0) 
            else:
                sum_ResNet8+=np.maximum(client.data_size/data_size_ResNet8 - a * client.d + b, 0) 
                
        self.p = {client.Client_id :  (np.maximum(client.data_size/data_size_MobileNetV2 - a * client.d + b, 0)/sum_MobileNetV2 if client.Client_model=='MobileNetV2' else  np.maximum(client.data_size/data_size_ResNet8 - a * client.d + b, 0)/sum_ResNet8 )for  client in clients}
    '''

