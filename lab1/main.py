from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

import ssl

from ffn import ffn_model
from cnn import cnn_model

ssl._create_default_https_context = ssl._create_unverified_context

# Завантаження датасету CIFAR100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Нормалізація даних
x_train, x_test = x_train / 255.0, x_test / 255.0

# Перетворення міток класів у one-hot encoded вектори
y_train, y_test = to_categorical(y_train, 100), to_categorical(y_test, 100)

# Навчання моделі FFN
ffn_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Навчання моделі CNN
cnn_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

