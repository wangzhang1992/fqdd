import torch  
from   torch.autograd import Variable  
from collections import OrderedDict 

#add learning_rate,epoch_n,loss

def init_nn():
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
#   w1 = Variable(torch.randn(input_data,hiddern_layer),requires_grad=True)
#   w2 = Variable(torch.randn(hiddern_layer,out_put),requires_grad=True)
    model = torch.nn.Sequential(OrderedDict([
         ("Line1", torch.nn.Linear(input_data,hiddern_layer)),
         ("ReLU1", torch.nn.ReLU()),
         ("Line2", torch.nn.Linear(hiddern_layer,out_put))
         ])
         )
    print(model)
    for epoch in range(epoch_n):
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        if epoch%1000 ==0:
           #print(loss)
           print("Epoch:{},Loss:{:.4f}".format(epoch,loss.data))
        model.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.data -= param.grad.data*learning_rate
init_nn() 
