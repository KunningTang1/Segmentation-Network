import torch
import torchvision.transforms as tfs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from uresnet import uresnet
from skimage import io

net = uresnet()
checkpoint = torch.load('F:\\python_Code\\Segmentation_code\\Example\\checkpoint\\19.pt')
net.load_state_dict(checkpoint['net_state_dict'])

size = 1200;size1 = 2600
def crop(data,height=size, width=size1):
    st_x = 0
    st_y = 0
    box = (st_x, st_y, st_x+width, st_y+height)
    data = data.crop(box)
    return data

im_tfs = tfs.Compose([
      tfs.ToTensor(),
      # tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])

net.eval()
img = io.imread('F:\\python_Code\\Segmentation_code\\Example\\Testing_image.png')
img = Image.fromarray(img)
cut_image = crop(img).convert('RGB')
norm_image = im_tfs(cut_image)
test_image = norm_image.unsqueeze(0).float()
out = net(test_image)
pred = out.max(1)[1].squeeze().data.cpu().numpy()
pred = np.uint8(pred)


plt.subplot(121)
plt.imshow(cut_image)
plt.title('Greyscale image')
plt.subplot(122)
plt.imshow(pred)
plt.title('Segmented image')

