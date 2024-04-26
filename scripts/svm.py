import torch
import torch.nn.functional as f
import numpy as np

class SVM(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        
        super(SVM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.line1 = torch.nn.Linear(input_dim,output_dim)

    def forward(self, x):

        x = torch.sigmoid(self.line1(x))
        return x


def dev(svm, dev_data):

    svm.eval()
    x_dev = dev_data[0]
    y_dev = dev_data[1]
    for i, x in enumerate(x_dev):
        print(x)
        output = svm(x)
        print('pre:{}\tref:{}'.format(output,y_dev[i]))

def train(svm, train_data, dev_data, optimer, epoch_n):
    
    x_train = train_data[0]
    y_train = train_data[1]
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epoch_n):
        svm.train()
        for i, x in enumerate(x_train):
            print(x)
            output = svm(x)
            optimer.zero_grad()
            print(y_train[i])
            loss = loss_fn(output,y_train[i])
            print(loss)
            loss.backward() 
            optimer.step()
    dev(svm, dev_data)
   
def main():
    
    torch.manual_seed(2)
    x_train = torch.randn(1000,5)
    y_train = torch.tensor(np.random.randint(0,2,1000).reshape(1000,1).astype(np.float32))
    x_dev = torch.randn(20,5)
    y_dev = torch.tensor(np.random.randint(0,2,20).reshape(20,1).astype(np.float32))
    train_data = (x_train, y_train)
    dev_data = (x_dev, y_dev)
    svm = SVM(5,1)
    optimer = torch.optim.Adam(svm.parameters(), lr=0.1)
    train(svm, train_data, dev_data, optimer, 200)

main()
