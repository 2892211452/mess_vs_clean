from torchvision import models







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
            nn.Conv2d(16,32,5,1,2),#(32,32,32)
            nn.ReLU(),#(32,32,32)
            nn.MaxPool2d(2)#(32,16,16)
        )
        self.dropout = nn.Dropout(0.1)


        #分类器
        self.classifier = nn.Sequential(
            nn.Linear(32*16*16,1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1000,100),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.Linear(100,num_classes)
        )

    #定义前向传播过程，过程名字不可更改，因为这是重写父类的方法
    def forward(self,x):
        x = self.conv1( x )
        x = self.normal(x)
        x = self.conv2( x ) #(batch,32,16,16)
        x= self.dropout(x)
        x=x.view(x.size(0),-1) #(batch,32*7*7)
        output=self.classifier(x)
        return output

    #传入图片
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
    
    #传入数组
def preNum(model, img):
            device = torch.device('cpu')
            ans = model(img)
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
            temp ,pro= preNum(model,temp)
            #print(temp, pro)

            device = torch.device('cpu')
            ans = Tlabels[i]
            ans = ans.to(device)
            index = ans.argmax()
            
            index = index.numpy()
           # print(index== temp )
            if index ==temp:
                count= count+1
        
        print('当前测试集的正确率')64).to(device))
        # print(y1)
        # print(labels[5])
        ac = count / Tlabels.size(0)
        return ac


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 来判断是否有可用的GPU

if  __name__ == "__main__":
    vgg = CNN(2).to(device)
    print(vgg)

    Vggmodel = models.vgg11(pretrained=True)

    #禁止求导,让数据不更新
    for parma in Vggmodel.parameters():
        parma.requires_grad = False


    Vggmodel.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2)
                                       )
        
    Vggmodel =Vggmodel.to(device)

    print(Vggmodel)






    print(vgg)

    fig = plt.figure(1, figsize=(5,5))
    ax=fig.add_subplot()
    #plt.ion()

    loss_F = nn.MSELoss()
    optimizer = torch.optim.Adam(Vggmodel.parameters(), lr=0.001, weight_decay=0.01)#最后一个参数设置正则化,防止过拟合



    images = []
    labels = []
    images, labels= load_dataset(filePath+"/images/train", images, labels)
    images= images/256
    images = torch.tensor(images).float().to(device)
    labels = torch.tensor(labels.astype(float)).to(device)

    images = images.view(images.size(0), 3, 64, 64)
    train = True

    #绘图所用的值
    x=[]
    y=[]
    Ty=[]

    if train:    
        images= Variable(images)
        labels= Variable(labels).float()

      
        for step in range(100):

            pre = Vggmodel(images.float())

            loss  = loss_F(pre, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1 ==0 :
                print("the step " +str(step) +" the loss is"+ str(loss))
                ac = getAc(Vggmodel)
                # time.sleep(1)
                print(ac)
            Ty.append(ac)
            
            x.append(step)
            y.append(loss)

            plt.cla()
            #ax.plot(x,y)
            ax.plot(x,Ty)

            plt.pause(0.003)

            
            Vggmodel.eval()

        torch.save(Vggmodel, filePath+'/mess_clean_model.h5') 
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


        



