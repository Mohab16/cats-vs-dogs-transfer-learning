import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from src.PyTorch.model import CatsDogsModel
from src.PyTorch.dataset import load_dataset

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def train_head_model():
    
    #Load data
    train_loader, validation_loader, class_names = load_dataset()
    
    #Create the model
    model = CatsDogsModel()
    model = model.to(DEVICE)
    
    #Loss
    criterion = nn.BCEWithLogitsLoss()
    
    #Optimizer
    
    optimizer = Adam(
        filter(
            #To ignore pretrained model layers 
            lambda parameter: parameter.requires_grad,
            model.parameters()
        ),
        lr=0.001
    )
    
    #Training
    epochs=5
    best_val_loss = float("inf")
    patience = 3
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(
                outputs.squeeze(1),
                labels
            )
            
            loss.backward()
            
            running_loss += loss.item()
            optimizer.step()
            predictions = (torch.sigmoid(outputs.squeeze(1)) >=0.5)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total   
        
        #Validation
        
        model.eval()
        
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(DEVICE)
                labels = labels.float().to(DEVICE)
                
                outputs = model(images)
                loss = criterion(
                    outputs.squeeze(1),
                    labels
                )
                running_val_loss += loss.item()
                
                predictions = (
                    torch.sigmoid(outputs.squeeze(1)) >=0.5
                )
                
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = running_val_loss / len(validation_loader)
        val_accuracy = val_correct / val_total
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                "../models/head_training.pth"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Accuracy: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Accuracy: {val_accuracy:.4f}"
        )
                
                
    #Restore best weights 
    model.load_state_dict(
        torch.load(
            "../models/head_training.pth"
        )
    )            
    return model              
    


def fine_tune_model():
    train_loader, validation_loader, class_names = load_dataset()
    
    model = CatsDogsModel()
    model.load_state_dict(
        torch.load("../models/head_training.pth")
    )
    model = model.to(DEVICE)
    
    for parameter in model.base_model.features.parameters():
        parameter.requires_grad = False
        
    for parameter in model.base_model.features[12:].parameters():
        parameter.requires_grad = True
    criterion = nn.BCEWithLogitsLoss()
        
    optimizer = Adam(
        filter(
         lambda parameter: parameter.requires_grad,
         model.parameters()   
        ),        
        lr= 0.00005,

    )
    #Training
    epochs=9
    best_val_loss = float("inf")
    patience = 3
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(
                outputs.squeeze(1),
                labels
            )
            
            loss.backward()
            
            running_loss += loss.item()
            optimizer.step()
            predictions = (torch.sigmoid(outputs.squeeze(1)) >=0.5)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total   
        
        #Validation
        
        model.eval()
        
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(DEVICE)
                labels = labels.float().to(DEVICE)
                
                outputs = model(images)
                loss = criterion(
                    outputs.squeeze(1),
                    labels
                )
                running_val_loss += loss.item()
                
                predictions = (
                    torch.sigmoid(outputs.squeeze(1)) >=0.5
                )
                
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = running_val_loss / len(validation_loader)
        val_accuracy = val_correct / val_total
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                "../models/fine_tuning.pth"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Accuracy: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Accuracy: {val_accuracy:.4f}"
        )
                
                
    #Restore best weights 
    model.load_state_dict(
        torch.load(
            "../models/fine_tuning.pth"
        )
    )            
    return model        

# def get_callbacks(FILE_PATH):
#     callbacks=[tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=3,
#     restore_best_weights=True
#     ),
#     tf.keras.callbacks.ModelCheckpoint(
#     filepath=FILE_PATH,
#     mode="min",
#     monitor="val_loss",
#     save_best_only=True,
#     verbose=1
#     )

#     ]
#     return callbacks



# def compile_model(model, learning_rate):
#     model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate),
#     loss="binary_crossentropy",
#     metrics=['accuracy']
#     )

# def train_head_model():
#     """
#     Train only the custom classification head.

#     During this stage, the pretrained MobileNetV2 backbone
#     remains frozen while only the newly added classifier
#     layers are updated.

#     Returns:
#         tuple:
#             model (tf.keras.Model):
#                 Trained model.

#             history (tf.keras.callbacks.History):
#                 Training history.
#     """

#     train_dataset, validation_dataset,_=load_dataset()
    
#     model=create_model()
#     compile_model(model, 0.001)
#     head_history=model.fit(train_dataset, epochs=5, validation_data=validation_dataset, callbacks=get_callbacks("../models/head_training.keras"))
#     return model,head_history

# def fine_tune_model(initial_epoch):
#     """
#     Fine-tune the pretrained MobileNetV2 backbone.

#     The pretrained model is loaded from disk, the top layers
#     are unfrozen, and training resumes using a smaller
#     learning rate.

#     Args:
#         initial_epoch (int):
#             Epoch from which fine-tuning should continue.

#     Returns:
#         tuple:
#             model (tf.keras.Model):
#                 Fine-tuned model.

#             history (tf.keras.callbacks.History):
#                 Fine-tuning history.
#     """
#     model=tf.keras.models.load_model('../models/head_training.keras')
#     base_model=model.get_layer("mobilenetv2_1.00_160")
#     train_dataset, validation_dataset,_=load_dataset()
    
#     base_model.trainable=True
#     fine_tune_at=120

#     for layer in base_model.layers[:fine_tune_at]:
#         layer.trainable=False
#     compile_model(model, 0.00005)
#     final_history=model.fit(train_dataset, epochs=10, initial_epoch=initial_epoch,validation_data=validation_dataset,callbacks=get_callbacks("../models/final_training.keras"))
#     return model, final_history


