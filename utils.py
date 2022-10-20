import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np

# Visualize single image
def show_image(X):
  if len(X.shape) == 4:
    X = X[0, :, :, :]
  image = rearrange(X, "c h w -> h w c")
  fig = plt.figure(figsize=(10, 10))
  plt.imshow(image)
  plt.show()

# Visualize grid of images
def show_images(X, y=None, nrow=3, ncol=3, randomize=False):
  b, c, h, w = X.shape
  if randomize:
    idxs = np.random.randint(0, b, nrow * ncol)
  else:
    idxs = np.arange(nrow * ncol)
  images = rearrange(X[idxs, :, :, :].numpy(), "b c h w -> b h w c")

  if y is None:
    labels = ["" for i in range(nrow * ncol)]
  else:
    labels = y[idxs].numpy()
  fig = plt.figure(figsize=(10, 10))

  for i in range(1, nrow * ncol + 1):
      fig.add_subplot(nrow, ncol, i)
      plt.imshow(images[i - 1, :, :, :])
      plt.xticks([])
      plt.yticks([])
      plt.title("{}".format(labels[i - 1]))
  plt.show()