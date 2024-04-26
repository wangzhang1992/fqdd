import torch  
import sys,os
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable  
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# torchvision包的主要功能是实现数据的处理，导入和预览等

# add # torch.nn中的类来搭建卷积神经网络
def init_nn():
    torch.manual_seed(2020)
    batch_n   =100  
    hiddern_layer=10
    input_data=1000
    out_put = 10
    epoch_n = 10000
    learning_rate = 1e-4
    #loss_fn = torch.nn.MSELoss() #torch.nn.MSELoss类使用均方误差函数对损失值进行计算，在定义类的对象时不用传入任何参数，但在使用实例时需要输入两个维度一样的参数方可进行计算
    #loss_fn = torch.nn.L1Loss()类使用平均绝对误差函数对损失值进行计算，同样，在定义类的对象时不用传入任何参数，但在使用实例时需要输入两个维度一样的参数进行计算.
    
    loss_fn = torch.nn.L1Loss()
    #x = Variable(torch.randn(100,100))
    #y = Variable(torch.randn(100,100))
    #loss = loss_f(x,y)
    #loss_fn = torch.nn.CrossEntropyLoss()#类用于计算交叉熵，在定义类的对象时不用传入任何参数，在使用实例时需要输入两个满足交叉熵的计算条件的参数
    x = Variable(torch.randn(batch_n,input_data),requires_grad=False)
    y = Variable(torch.randn(batch_n,out_put),requires_grad=False)
    #Variable的梯度置为0: Variable.grad.data.zero_()
#   w1 = Variable(torch.randn(input_data,hiddern_layer),requires_grad=True)
#   w2 = Variable(torch.randn(hiddern_layer,out_put),requires_grad=True)
    asrdnn = torch.nn.Sequential(OrderedDict([
         ("Line1", torch.nn.Linear(input_data,hiddern_layer)),
         ("ReLU1", torch.nn.ReLU()),
         ("Line2", torch.nn.Linear(hiddern_layer,out_put))
         ])
         )
    print(asrdnn)
    #optimzer = torch.optim.Adam(asrdnn.parameters(), lr = learning_rate)
    #optimzer = torch.optim.SGD(asrdnn.parameters(), lr = learning_rate)
    optimzer = torch.optim.RMSprop(asrdnn.parameters(), lr = learning_rate)
    #optimzer = torch.optim.Adagrad(asrdnn.parameters(), lr = learning_rate)
    for epoch in range(epoch_n):
        y_pred = asrdnn(x)
        loss = loss_fn(y_pred,y)
        if epoch%1000 ==0:
           #print(loss)
           print("Epoch:{},Loss:{:.4f}".format(epoch,loss.data))
        asrdnn.zero_grad() #直接把模型的参数梯度设成0, method1: asrdnn.zero_grad(); method2: optimizer.zero_grad(); 
        loss.backward()
        #进行梯度更新
        optimzer.step()
init_nn()
