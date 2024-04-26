import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

fig = plt.figure()
trloss = fig.add_subplot(231)
xx = np.random.randint(20)
train_loss = np.random.randint(20)
trloss.set(xlim=[0, 20], title="train_loss",
           ylabel='loss', xlabel='epoch')

trloss.plot(xx, train_loss, color='darkred')
plt.show()
