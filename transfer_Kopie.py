from PIL import Image
import matplotlib
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import time
from io import BytesIO

matplotlib.use('TkAgg')
import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms, models

import HRNet
import utils
#'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('------------------------------------------------------------------')
print(device)
print('------------------------------------------------------------------')
#'''
#with torch.cuda.device(1)
#device = torch.cuda.set_device(1)
#print(torch.cuda.current_device())

# get the VGG19's structure except the full-connect layers
VGG = models.vgg19(pretrained=True).features
VGG.to(device)
print(VGG)
# only use VGG19 to extract features, we don't need to change it's parameters
for parameter in VGG.parameters():
    parameter.requires_grad_(False)

style_net = HRNet.HRNet()
style_net.to(device)
print(style_net)

content_image = utils.load_image("./Datasets/Baseline_Train_VOC/JPEGImages/0041222.jpg", img_size=800)  # temporary/content.png
content_image = content_image.to(device)

#style_image = utils.load_image("./Datasets/omnidetector-Flat/JPEGImages/Record_00714.jpg")  # temporary/style.png  Dataset_Omni
#style_image = utils.load_image("./Datasets/indoor_cvpr/Images/bathroom/bothroom99.jpg")  # temporary/style.png  Dataset_indoor
#style_image = utils.load_image("/home/veas/datasets/yt-fisheye/2MP_1080p_360_Panoramic_Fisheye_VR_Camera_Indoor_Room_Video-BwmZvRcx994/img_00050.jpg")  # temporary/style.png  Dataset_yt_fisheye_
style_image = utils.load_image("/home/veas/datasets/yt-fisheye/Fisheye_Security_Camera_Demo-_Knr1HuGypI/img_00050.jpg")  # temporary/style.png  Dataset_yt_fisheye_
#style_image = utils.load_image("/home/veas/datasets/yt-fisheye/6MP_Fisheye_Camera-ByejHxegG9Q/img_00050.jpg")  # temporary/style.png  Dataset_yt_fisheye_
#style_image = utils.load_image("/home/veas/datasets/yt-fisheye/Compact_Fisheye_Network_Camera__-__DS-2CD2942F-IS_1.9mm-dwzQH7IAW34/img_00031.jpg")  # temporary/style.png  Dataset_yt_fisheye_
#style_image = utils.load_image("/home/veas/datasets/yt-fisheye/Fisheye_Security_Camera_Demo-_Knr1HuGypI/img_00050.jpg")  # temporary/style.png  Dataset_yt_fisheye_
style_image = style_image.to(device)

# display the raw images
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
#ax1.imshow(utils.im_convert(content_image))
#ax2.imshow(utils.im_convert(style_image))
#plt.show()

content_features = utils.get_features(content_image, VGG)
style_features = utils.get_features(style_image, VGG)

style_gram_matrixs = {layer: utils.get_grim_matrix(style_features[layer]) for layer in style_features}

target = content_image.clone().requires_grad_(True).to(device)

# try to give fore con_layers more weight so that can get more detail in output iamge
style_weights = {'conv1_1': 0.1,
                 'conv2_1': 0.2,
                 'conv3_1': 0.4,
                 'conv4_1': 0.8,
                 'conv5_1': 1.6}

content_weight = 150
style_weight = 1

show_every = 100
#show_every = 1
optimizer = optim.Adam(style_net.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
steps = 5000
#steps = 1

content_loss_epoch = []
style_loss_epoch = []
total_loss_epoch = []
output_image = content_image

time_start = time.time()
for epoch in range(0, steps + 1):

    scheduler.step()
#########################################################
    target = style_net(content_image).to(device)
    target.requires_grad_(True)

    target_features = utils.get_features(target, VGG)  # extract output image's all feature maps
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    style_loss = 0
###########################################################

    # compute each layer's style loss and add them
    for layer in style_weights:
        target_feature = target_features[layer]  # output image's feature map after layer
        target_gram_matrix = utils.get_grim_matrix(target_feature)
        style_gram_matrix = style_gram_matrixs[layer]

        layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
        b, c, h, w = target_feature.shape
        style_loss += layer_style_loss / (c * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss_epoch.append(total_loss)

    style_loss_epoch.append(style_weight * style_loss)
    content_loss_epoch.append(content_weight * content_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % show_every == 0:
        print("After %d criterions:" % epoch)
        #print("After %d criterions:" % steps)
        print('Total loss: ', total_loss.item())
        print('Content loss: ', content_loss.item())
        print('Style loss: ', style_loss.item())
        #plt.imshow(utils.im_convert(target))
        #plt.show()

    output_image = target
time_end = time.time()
print('totally cost', time_end - time_start)
#output_image.savefig("./Datasets/STYLED_IMAGES/".format(epochs))
#plt.close()
# plot the line chart

torch.save(style_net.state_dict(),'StyleTransferModel.pth')

epoch = range(0, steps + 1)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

ax1.plot(epoch, total_loss_epoch)
ax1.set_title("Total loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("Total loss")

ax2.plot(epoch, style_loss_epoch)
ax2.set_title("Style loss")
ax2.set_xlabel("epoch")
ax2.set_ylabel("Style loss")

ax3.plot(epoch, content_loss_epoch)
ax3.set_title("Content loss")
ax3.set_xlabel("epoch")
ax3.set_ylabel("Content loss")

plt.show()

# display the raw images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#content and style ims side-by-side
ax1.imshow(utils.im_convert(content_image))
ax2.imshow(utils.im_convert(output_image))
plt.show()
image = utils.im_convert(output_image)
img2 = Image.open('style.png')      #opening style.png image from original code
print(type(img2))                #class 'PIL.PngImagePlugin.PngImageFile'
print(type(image))              #class 'numpy.ndarray'
print(type(output_image))       #class 'torch.Tensor'
#img = Image.open(BytesIO(image))
#image = Image.fromarray(np.uint8(cm.gist_earth(image)*255))
#plt.imsave("./Datasets/STYLED_IMAGES/{}.jpg".format(steps),image,cmap = 'Greys')
#image.save("./Datasets/STYLED_IMAGES/{}.jpg".format(epoch))
#image.save("./Datasets/STYLED_IMAGES/{}.jpg".format(epoch))
#plt.close()

'''
checkpoint = {'model': style_net(),
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()
          }

torch.save(checkpoint, 'checkpoint.pth')
'''

#model = target
#torch.save(model.state_dict(),"/home/veas/Projects/PycharmProjects/MASTER_THESIS/StyleTransfer/")
#torch.save(model.state_dict(),"./Datasets/Models/model/model.state_dict")
#torch.save(model,"./Datasets/Models/model/model.h5")
#torch.save(model.state_dict(),"./Datasets/Models/model/model")
#torch.save(model.state_dict(),'StyleTransferModel.pth')



