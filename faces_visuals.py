
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.utils import shuffle




if __name__ == '__main__':
    
    # load the face dataset
    dataset = np.load('5-celebrity-faces-dataset.npz')
    trainX, trainy, testX, testy = dataset['arr_0'], dataset['arr_1'], dataset['arr_2'], dataset['arr_3']
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    fig , axs = plt.subplots(nrows=2,ncols=3,figsize=(10,10))
    img_index = 0

    # shuffle the dataset
    trainX, trainy = shuffle(trainX,trainy,random_state=42)

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            axs[i,j].set_xlabel(trainy[img_index])
            axs[i,j].imshow(trainX[img_index])
            img_index += 1
    plt.show()