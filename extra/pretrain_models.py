import os
import lab
import torch
import torch.nn as nn
import torchvision

# Save a copy of AlexNet
model = torchvision.models.alexnet(pretrained=True)
torch.save(model.state_dict(), os.path.join('data','alexnet.pth'))

def new_model():
    return nn.Sequential(
        nn.Conv2d(1,20,5,bias=True),
        nn.MaxPool2d(2,stride=2),
        nn.Conv2d(20,50,5,bias=True),
        nn.MaxPool2d(2,stride=2),
        nn.Conv2d(50,500,4,bias=True),
        nn.ReLU(),
        nn.Conv2d(500,26,2,bias=True),
    )

# Load data
imdb = torch.load('data/charsdb.pth')
im_mean = imdb['images'].mean()
imdb['images'].sub_(im_mean)

# Train model without jitter
model = new_model()
lab.train_model(model, imdb)
torch.save(model.state_dict(), os.path.join('data','charscnn.pth'))

# Train model with jitter
model_jitter = new_model()
lab.train_model(model_jitter, imdb, jitter=lab.jitter)
torch.save(model_jitter.state_dict(), os.path.join('data','charscnn_jitter.pth'))
