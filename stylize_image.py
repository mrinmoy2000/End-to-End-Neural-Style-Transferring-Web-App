import re

import torch
from torchvision import transforms
from PIL import Image
import torch.onnx
from fast_neural_style.transformer_net import TransformerNet


# Loading the image
def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


# Saving the image
def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


# Setting the device to the gpu if it is available otherwise using the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loading the model
def load_model(model_path):
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model_path)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        return style_model


# Stylizing the image
def stylize(style_model, content_image, output_image):
    content_image = load_image(content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(225))
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = style_model(content_image).cpu()

    save_image(output_image, output[0])
