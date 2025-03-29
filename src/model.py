from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
import logging

from src.config import N_FEATURES, N_CLASSES, LEARNING_RATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_cnn_model(input_shape=(N_FEATURES, 1), num_classes=N_CLASSES):
    """Builds a 1D Convolutional Neural Network model."""
    model = Sequential(name="ECG_CNN_Classifier")

    # Input Layer
    model.add(Input(shape=input_shape, name='input_layer'))

    # Convolutional Block 1
    model.add(Conv1D(filters=64, kernel_size=5, strides=1, padding='same', name='conv1d_1'))
    model.add(BatchNormalization(name='batchnorm_1'))
    model.add(Activation('relu', name='relu_1'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same', name='maxpool_1'))
    model.add(Dropout(0.2, name='dropout_1'))

    # Convolutional Block 2
    model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding='same', name='conv1d_2'))
    model.add(BatchNormalization(name='batchnorm_2'))
    model.add(Activation('relu', name='relu_2'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same', name='maxpool_2'))
    model.add(Dropout(0.3, name='dropout_2'))

    # Convolutional Block 3
    model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', name='conv1d_3'))
    model.add(BatchNormalization(name='batchnorm_3'))
    model.add(Activation('relu', name='relu_3'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same', name='maxpool_3'))
    model.add(Dropout(0.3, name='dropout_3'))

    # Flatten
    model.add(Flatten(name='flatten'))

    # Dense Block
    model.add(Dense(256, name='dense_1'))
    model.add(BatchNormalization(name='batchnorm_dense_1'))
    model.add(Activation('relu', name='relu_dense_1'))
    model.add(Dropout(0.4, name='dropout_dense_1'))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax', name='output_layer'))

    # Compile Model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    logging.info("CNN model built successfully.")
    model.summary(print_fn=logging.info) # Log model summary

    return model