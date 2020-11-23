#!/usr/bin/env python3

from torchvision import models
import numpy as np
import sys
import os
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
from autocrop import Cropper

def decode_segmap(image, source, nc = 21):
    label_colours = np.array([(0, 0, 0), 
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colours[l, 0]
        g[idx] = label_colours[l, 1]
        b[idx] = label_colours[l, 2]

    rgb = np.stack([r, g, b], axis = 2)

    foreground = cv2.imread(source)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))

    background = 255 * np.ones_like(rgb).astype(np.uint8)

    foreground = foreground.astype(float)
    background = background.astype(float)

    th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)

    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)

    alpha = alpha.astype(float) / 255

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)

    outImage = cv2.add(foreground, background)

    return outImage / 255

def segment(model, image):
    cropper = Cropper()
    cropped_array = cropper.crop(image)
    img = Image.fromarray(cropped_array)
    
    idx = image.index('.')
    name = 'temp' + image[idx:]
    img.save(name)

    plt.imshow(img); plt.axis('off'); plt.show()

    trf = T.Compose([T.Resize(256),
                     T.CenterCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)
    out = model(inp)['out']
    om = torch.argmax(out.squeeze(), dim = 0).detach().cpu().numpy()
    rgb = decode_segmap(om, name)

    if os.path.isfile(name):
        os.remove(name)

    plt.imshow(rgb); plt.axis('off'); plt.show()

if len(sys.argv) != 2:
    print('Enter command line argument')
    sys.exit(0)

# fcn = models.segmentation.fcn_resnet101(pretrained = True).eval()
dlab = models.segmentation.deeplabv3_resnet101(pretrained = 1).eval()

segment(dlab, sys.argv[1])
