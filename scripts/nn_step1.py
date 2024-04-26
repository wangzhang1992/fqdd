import torch
from torch.autograd import Variable
from collections import OrderedDict

def init_nn(): 
    batch_n = 100  #batch size
    hidden_layer = 100 #output feature dim by passing hidden layer
    input_data = 1000 #input data's dim
    output_data = 10 #finally output result'dim by classfiying
 
    models = torch.nn.Sequential(OrderedDict([
           ("Linel",torch.nn.Linear(input_data,hidden_layer)),
           ("ReLU1",torch.nn.ReLU()),
           ("Line2",torch.nn.Linear(hidden_layer,output_data))
           ])
           )
    print(models)
init_nn()
