import numpy as np
import matplotlib.pyplot as plt

def plot_grid(data, targets, class_num, grid_size=4, trs=None):
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    ax = np.ravel(ax)

    samples = data[targets == class_num]
    idxs = np.arange(len(samples))
    samples = samples[np.random.choice(idxs, size=grid_size**2, replace=False)]

    for img, ax in zip(samples, ax):
        if trs is not None:
            img = trs(image=img)['image'].numpy()
        ax.imshow(img)
        ax.axis('off')
    
    plt.show()