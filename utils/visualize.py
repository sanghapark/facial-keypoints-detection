import numpy as np
import matplotlib.pyplot as plt

def show_keypoints_on_data(X, Y):
    def plot_sample(x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10, c='r')

    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(X.shape[0]):
        ax = fig.add_subplot(int(np.sqrt(X.shape[0])), int(np.sqrt(Y.shape[0])), i + 1, xticks=[], yticks=[])
        plot_sample(X[i], Y[i], ax)

    plt.show()