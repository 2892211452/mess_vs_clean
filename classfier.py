import torch as torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt #作图用到的库
import time
from torch.autograd import Variable
from load_data  import *
import os
import time
filePath=  os.path.dirname(__file__) #获取当前文件所在目录

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(          #(3,64,64)
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2   #padding=(kernelsize-stride)/2
            ),#(16,64,64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#(16,32,32)
 
        )
        
        self.normal = nn.BatchNorm2d(16) # 数据归一化,防止过拟合
        self.conv2=nn.Sequential(#(16,32,32)
            nn.Conv2d(16,16,5,1,2),#(16,32,32)
            nn.ReLU(),#(16,32,32)
        )
        self.conv3=nn.Sequential(#(16,32,32)
            nn.Conv2d(16,16,5,1,2),#(16,32,32)
            nn.ReLU(),#(16,32,32)
        )
        self.conv4=nn.Sequential(#(16,32,32)
            nn.Conv2d(16,32,5,1,2),#(32,32,32)
            nn.ReLU(),#(32,32,32)
            nn.MaxPool2d(2)#(32,16,16)
        )
        self.dropout = nn.Dropout(0.1)


        #分类器
        self.classifier = nn.Sequential(
            nn.Linear(32*16*16,1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2, inplace=False),

            nn.Linear(1000,100),
            nn.ReLU(True),
            nn.Dropout(0.5, inplace=False),

            nn.Linear(100,num_classes),
            nn.LogSoftmax(dim=1)
        )

    #定义前向传播过程，过程名字不可更改，因为这是重写父类的方法
    def forward(self,x):
        x = self.conv1( x )
        x= self.conv2(x)
        x= self.conv3(x)
        x = self.normal(x)
        x = self.conv4( x ) #(batch,32,16,16)
        x= self.dropout(x)
        x=x.view(x.size(0),-1) #(batch,32*7*7)
        output=self.classifier(x)
        return output


    def preImg(self, img):
            image = resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
            image = np.array(image)
            image = image/256
            image= torch.tensor(image).view(-1, 3,64,64).float()
            ans = self.forward(image)
            index = ans.argmax()
            
            index = index.numpy()
            ans = ans.view(ans.size(1))
            pro = ans.detach().numpy()[index]
            #print(ans)

            return index, pro
    
    def preNum(self, img):
            device = torch.device('cpu')
            ans = self.forward(img)
            ans = ans.to(device)
            index = ans.argmax()
            
            index = index.numpy()
            ans = ans.view(ans.size(1))
            pro = ans.detach().numpy()[index]
            #print(ans)

            return index, pro


def getAc(model):
        device = torch.device('cuda')

        model = model
        Timages = []
        Tlabels =[]

        Timages, Tlabels= load_dataset(filePath+"/images/val", Timages, Tlabels)

        Timages= Timages/256
        Timages = torch.tensor(Timages).float().to(device)
        Tlabels = torch.tensor(Tlabels.astype(float)).to(device)

        Timages = Timages.view(Timages.size(0), 3, 64, 64)
        count = 0

        for i in range(Timages.size(0)):
            temp = Timages[i].view(-1, 3,64,64)
            temp ,pro= model.preNum(temp)
            #print(temp, pro)

            device = torch.device('cpu')
            ans = Tlabels[i]
            ans = ans.to(device)
            index = ans.argmax()
            
            index = index.numpy()
           # print(index== temp )
            if index ==temp:
                count= count+1
        
        print('当前测试集的正确率')
        ac = count / Tlabels.size(0)
        return ac


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 来判断是否有可用的GPU

if  __name__ == "__main__":
    vgg = CNN(2).to(device)
    print(vgg)

    fig = plt.figure(1, figsize=(5,5))
    ax=fig.add_subplot()
    #plt.ion()

    loss_F = nn.MSELoss()
    optimizer = torch.optim.Adam(vgg.parameters(), lr=0.001, weight_decay=0.01)#最后一个参数设置正则化,防止过拟合



    images = []
    labels = []
    images, labels= load_dataset(filePath+"/images/train", images, labels)
    images= images/256
    images = torch.tensor(images).float()
    labels = torch.tensor(labels.astype(float))
    

    images = images.view(images.size(0), 3, 64, 64)


    torch_dataset = Data.TensorDataset(images, labels) #将x,y读取，转换成Tensor格式
    loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=30,      # 最新批数据
        shuffle=True,               # 是否随机打乱数据
        num_workers=2,              # 用于加载数据的子进程
    )


    train = True

    #绘图所用的值
    x=[]
    y=[]
    Ty=[]

    if train:    
        images= Variable(images)
        labels= Variable(labels).float()

      
        for step in range(200):
            for index, (batch_x, batch_y) in enumerate(loader):  # 每个训练步骤
                batch_x =batch_x.to(device)
                batch_y =batch_y.to(device).float()
                pre = vgg(batch_x.float())

                loss  = loss_F(pre, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 3 ==0 :
                    print("the step " +str(step) +" the loss is"+ str(loss))
                    ac = getAc(vgg)
                    # time.sleep(1)
                    print(ac)
                    Ty.append(ac)
                    
                    x.append(step)
                    y.append(loss)

                plt.cla()
                #ax.plot(x,y)
                ax.plot(x,Ty)

                plt.pause(0.003)

                
                vgg.eval()

        torch.save(vgg, filePath+'/mess_clean_model.h5') 
    else:


        #cpu进行单个测试
   
        # device = torch.device('cpu')
        # model = CNN(2)
        # model = torch.load(filePath+'/mess_clean_model.h5', map_location=device)
        # y1= model(images[5].view(-1, 3,64,64).to(device))
        # print(y1)
        # print(labels[5])
        # image = cv2.imread(filePath+'/images/train/clean/2.png')

        # print(model.preImg(image))



        #进行测试集测试

        model = torch.load(filePath+'/mess_clean_model.h5')
        ac = getAc(model)
        print(ac)


        



