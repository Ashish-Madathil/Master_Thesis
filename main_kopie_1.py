
import numpy as np
from PIL import  Image
from glob import glob
import torch
import os
import utils
import torch.nn
import torch.optim as optim
from torchvision import transforms, models
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import HRNet
from tqdm import tqdm
#import transfer_Kopie

style_net = HRNet.HRNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('------------------------------------------------------------------')
print(device)
print('------------------------------------------------------------------')

#model = transfer_Kopie.save('StyleTransferModel.pth')
model = HRNet.HRNet()
model.load_state_dict(torch.load('StyleTransferModel.pth'))
model.eval()

model = model.to(device)
#filenames = np.array(glob("./Datasets/Train_VOC/JPEGImages/*.jpg"))

def transfer_image(image):
  #  image_tensor = test_transforms(image).float()
    input = utils.load_image(image)
    input = input.to(device)
    output = model(input)
    output_img = (utils.im_convert(output) * 255.0).astype(np.uint8)
    #output = StyleTransferModel(input)
    return output_img

for image in tqdm(glob("./Datasets/Baseline_Train_VOC/JPEGImages/*.jpg")):
    image_name = os.path.basename(image)
    output = transfer_image(image)
    #Image.fromarray(output).save("./Datasets/Dataset_1_Omni/{}".format(image_name))
    #Image.fromarray(output).save("./Datasets/Dataset_2_Indoor/{}".format(image_name))
    #Image.fromarray(output).save("./Datasets/Dataset_3_2MP_Panoramic_fisheye/{}".format(image_name))
    Image.fromarray(output).save("./Datasets/Dataset_4_Fisheye_Security_Camera/{}".format(image_name))
    #Image.fromarray(output).save("./Datasets/Dataset_5_6MP_Fisheye _Camera/{}".format(image_name))
    #Image.fromarray(output).save("./Datasets/Dataset_6_Compact_Fisheye_Network_Camera/{}".format(image_name))
    #Image.fromarray(output).save("./Datasets/Dataset_7_Fisheye_Security_Camera_Demo/{}".format(image_name))

print("dataset complete")

