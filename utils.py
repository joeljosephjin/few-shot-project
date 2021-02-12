import os
import matplotlib.image as mpimg
from skimage.transform import resize
import matplotlib.pyplot as plt
from random import randint
import torch
import numpy as np


def get_images_from(test_folder):
    eval_list=[]
    listing = os.listdir(test_folder)
    for imagename in listing:
        print(imagename)
        image = preprocess(test_folder+imagename)
        eval_list.append(image)
    evaluation_data = torch.cat(eval_list, dim=0)
    return evaluation_data

def preprocess(path):
    image = mpimg.imread(path)
    image = image[:,:,0]
    image = resize(image, (28,28))
    image = image.astype('float32')

    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    return image

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)
