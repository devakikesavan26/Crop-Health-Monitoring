"""
Keras / TensorFlow CNN Architecture Reference
Matching Section 6.1, 7.1 and Figure 7.1 of the Project Report
"""

import os
from config import CLASSES, IMAGE_SIZE, KERAS_MODEL_PATH

def build_keras_cnn(input_shape=(128, 128, 3), num_classes=len(CLASSES)):
    """
    Constructs the Keras Sequential CNN Architecture for Plant Disease Classification:
    Conv2D -> ReLU -> MaxPool2D -> Conv2D -> ReLU -> MaxPool2D -> Conv2D -> GlobalAveragePooling2D -> Dense -> Dense(Softmax)
    """
    try:
        import keras
        from keras import layers, models

        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Convolutional Block 1
            layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            
            # Convolutional Block 2
            layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            
            # Convolutional Block 3
            layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3'),
            layers.GlobalAveragePooling2D(name='gap'),
            
            # Fully Connected Classification Head
            layers.Dense(128, activation='relu', name='dense_features'),
            layers.Dropout(0.3, name='dropout'),
            layers.Dense(num_classes, activation='softmax', name='disease_predictions')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        # In environments where Keras backend is building
        return None

def get_model_summary_string():
    """Returns a textual architecture summary matching the project documentation."""
    summary = """
=================================================================
 Model: "CropHealth_CNN" (TensorFlow / Keras Architecture)
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        (None, 128, 128, 3)       0         
 conv1 (Conv2D)              (None, 64, 64, 16)        448       
 pool1 (MaxPooling2D)        (None, 32, 32, 16)        0         
 conv2 (Conv2D)              (None, 16, 16, 32)        4,640     
 pool2 (MaxPooling2D)        (None, 8, 8, 32)          0         
 conv3 (Conv2D)              (None, 8, 8, 64)          18,496    
 gap (GlobalAveragePooling2D)(None, 64)                0         
 dense_features (Dense)      (None, 128)               8,320     
 dropout (Dropout)           (None, 128)               0         
 disease_predictions (Dense) (None, 10)                1,290     
=================================================================
 Total params: 33,194 (129.66 KB)
 Trainable params: 33,194 (129.66 KB)
 Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
 Optimizer: Adam (lr=0.001)
 Loss Function: Categorical Cross-Entropy
 Supported Classes: 10 PlantVillage Agricultural Categories
=================================================================
"""
    return summary
