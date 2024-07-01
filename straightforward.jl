using MLDatasets, ImageShow, ImageIO

data = MNIST(:train)
(X, y) = data.features, data.targets
convert2image(MNIST, X[:, :, 1])