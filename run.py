import numpy as np
from utils import load_mnist_data, preprocess_data, accuracy

# 加载数据集
train_images, train_labels = load_mnist_data('train')
test_images, test_labels = load_mnist_data('test')

# 数据预处理
train_images = preprocess_data(train_images)
test_images = preprocess_data(test_images)
