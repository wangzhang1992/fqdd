import torch
import torch.nn as nn

class CNN(nn.Module):
      def __init__(self, output_dim):
          # (b, t, d) -> (b, 1, t, d), 对时间进行卷积，(b,256, t/2,d/128)
          super(CNN, self).__init__()
          self.output_dim = output_dim
          self.conv1 = nn.Sequential(
              nn.Conv2d(in_channels=1,
                        out_channels=16,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU()
          )

          self.conv2 = nn.Sequential(
              nn.Conv2d(16, 32, 3, 2, 1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()
          )

          self.conv3 = nn.Sequential(
              nn.Conv2d(32, 64, 3, 2, 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
          )

          self.line1 = nn.Sequential(nn.Linear(40, 1024), nn.ReLU(True))
          self.line2 = nn.Sequential(nn.Linear(1024, self.output_dim))

      def forward(self, x):
          x = torch.unsqueeze(x,1)
          x = self.conv1(x)
          x = self.conv2(x)
          x = self.conv3(x)
          x = self.line1(x.view(x.size(0), -1, 40))
          x = self.line2(x)
          return x

