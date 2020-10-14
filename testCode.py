from PIL import Image
import torchvision.transforms as T
import torchvision
import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random

img_path = '/content/test/ILSVRC2012_test_00000004.jpg'
#img_path = '/content/test/people.jpg'
normalImg = Image.open(img_path) # Load the image
dataset = ImageDataTest('/content/test/', '/content/test/test.lst')
data_loader = data.DataLoader(dataset=dataset, batch_size=1,  num_workers=30)
 
img_num = len(data_loader)
print(img_num)
for i, data_batch in enumerate(data_loader):
    #print('test')   
    images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
    with torch.no_grad():
        images = Variable(images)
        
        images = images.cpu()
        model.load_state_dict(torch.load('/content/PoolNet1.pth'))
        model.eval()
        print((images.size()))
        print(images)
        preds = model(images) #PMModel
        print(preds)
        pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
        multi_fuse = 65534 * pred #255
        cv2.imwrite(os.path.join( '1221.png'), multi_fuse)
        print(multi_fuse)


## for this part make folder inside test and put image in it
"""

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

valdir = '/content/test'

dataset_test = torchvision.datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

test_sampler = torch.utils.data.SequentialSampler(dataset_test)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1,
    sampler=test_sampler)

model.load_state_dict(torch.load('/content/PoolNet1.pth'))
model.eval()

with torch.no_grad():
    for image, target in data_loader_test:
        print(image)
        print(image.size())
        output = model(image)
        print(output)

##
"""
