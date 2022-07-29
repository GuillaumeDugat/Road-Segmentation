import numpy as np

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


# Code adopted from: https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
def get_embeddings(images):
    # Load the pretrained model and choose layer with vector representation
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    layer = model._modules.get('avgpool')

    model.eval()

    # need to normalize and resize each image to 224x224 pixels
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    features = []
    for image in images:
        tensor = to_tensor(image)[:3, :, :] # omit alpha channel
        tensor = scaler(tensor)
        img = Variable(normalize(tensor).unsqueeze(0))
        
        # create empty vector for features
        f = torch.zeros(512)
        
        # define a function that will copy the output of a layer
        def copy_data(m, i, o):
            f.copy_(o.data.reshape(o.data.size(1)))

        # attach that function to the selected layer
        h = layer.register_forward_hook(copy_data)

        # run resnet18 model on image
        model(img)

        # detach copy function from layer
        h.remove()

        features.append(np.array(f))

    return np.array(features)