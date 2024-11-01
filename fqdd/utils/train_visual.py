from visdom import Visdom
import time
import numpy as np


viz = Visdom(env='demo', server='0.0.0.0', port=6007, use_incoming_socket=False)
x, y = 0, 0
for i in range(50):
    x = i
    y = i * i
    viz.line(
        X=np.array([x]),
        Y=np.array([y]),
        win='window',
        update='append')
    time.sleep(5)
