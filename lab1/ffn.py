from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten

ffn_model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(100, activation='softmax')
])

# Компіляція моделі
ffn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])