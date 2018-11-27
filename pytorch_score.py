# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torchvision import transforms
import json
import base64
from io import BytesIO
from PIL import Image

from azureml.core.model import Model


def preprocess_image(image_file):
    """Preprocess the input image."""
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_file)
    image = data_transforms(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image


def base64ToImg(base64ImgString):
    base64Img = base64ImgString.encode('utf-8')
    decoded_img = base64.b64decode(base64Img)
    return BytesIO(decoded_img)


def init():
    global model
    model_path = Model.get_model_path('dog10')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()


def run(input_data):
    img = base64ToImg(json.loads(input_data)['data'])
    img = preprocess_image(img)

    # get prediction
    output = model(img)

    classes = ['Chihuahua',
            'Italian_greyhound',
            'whippet',
            'golden_retriever',
            'Shetland_sheepdog',
            'German_shepherd',
            'boxer',
            'Saint_Bernard',
            'malamute',
            'Siberian_husky']
    ## If you try with 20 classes please uncomment this:
#    classes =['Chihuahua',
#             'Italian_greyhound',
#             'whippet',
#             'Yorkshire_terrier',
#             'golden_retriever',
#             'Labrador_retriever',
#             'Shetland_sheepdog',
#             'Border_collie',
#             'German_shepherd',
#             'Bernese_mountain_dog',
#             'boxer',
#             'bull_mastiff',
#             'French_bulldog',
#             'Great_Dane',
#             'Saint_Bernard',
#             'Siberian_husky',
#             'basenji',
#             'pug',
#             'Samoyed',
#             'Pembroke'
    softmax = nn.Softmax(dim=1)
    pred_probs = softmax(model(img)).detach().numpy()[0]
    index = torch.argmax(output, 1)

    result = json.dumps({"label": classes[index], "probability": str(pred_probs[index])})
    return result