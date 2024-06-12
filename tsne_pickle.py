from tsnecuda import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

def load_data(file):

    print(file)
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
        # data = joblib.load(fo)
    return data

file_path = r""

dataset = load_data(file_path)
data = dataset['data']

labels = dataset['labels']

(x_train, y_train) = (data, labels)
print(y_train.shape)  # (n_samples,1)
print(x_train.shape)  # (n_samples,size*size*3)

tsne = TSNE(n_iter=1000, verbose=1, num_neighbors=32, device=0)
tsne_results = tsne.fit_transform(x_train)

print(tsne_results.shape)  # (n_samples,2)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, title='TSNE')


scatter = ax.scatter(
    x=tsne_results[:, 0],
    y=tsne_results[:, 1],
    c=y_train,
    # cmap=plt.cm.get_cmap('Paired'),
    # alpha=0.4,
    s=10)

legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend1)

plt.show()
plt.savefig('./tSNE.jpg')


