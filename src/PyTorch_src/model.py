import torch
import torch.nn as nn
from torchvision import models

from src.PyTorch.dataset import  IMG_SIZE



IMG_SHAPE=IMG_SIZE + (3,)

class CatsDogsModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Pretrained model
        self.base_model=models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        #Freeezing pretrained model layers
        for parameter in self.base_model.features.parameters():
            parameter.requires_grad = False
            
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.dropout = nn.Dropout(0.2)   
        
        self.classifier = nn.Linear(1280,1) 
        
    def forward(self, x):
        x = self.base_model.features(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x 
    

            
    
    
    
# def create_model(data_augmentation=data_augmenter()):
#     """
#     Build the Cats vs Dogs classifier using transfer learning.

#     A pretrained MobileNetV2 model is used as the feature extractor
#     with frozen weights. A custom classification head is added on top.
#     Returns:
#     tf.keras.Model:
#     Binary image classification model.
#     """

#     base_model = tf.keras.applications.MobileNetV2(
#     include_top=False,
#     weights="imagenet",
#     input_shape=IMG_SHAPE
#     )

#     base_model.trainable=False
#     inputs=tf.keras.Input(shape=IMG_SHAPE)
#     x=data_augmentation(inputs)
#     x=preprocess_input()(x)
#     x=base_model(x,training=False)
#     x=layers.GlobalAveragePooling2D()(x)
#     x=layers.Dropout(0.2)(x)

#     outputs=layers.Dense(1, activation='sigmoid')(x)
#     model=tf.keras.Model(inputs, outputs)

#     return model

