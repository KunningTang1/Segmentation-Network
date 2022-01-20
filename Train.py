import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
from uresnet import Uresnet
from utility import label_accuracy_score
from EN import efficientunet_b5
from unet import unet

# Hyperparameters define
batchsize = 8
EPOCH = 100
LEARNING_RATE = 0.00001
patch_size = 120
start_epoch_forsave = 40
save_interval = 2
learning_rate_decay = 0.5

net = efficientunet_b5()
# net = Uresnet()
if torch.cuda.is_available():
    net = net.cuda()  
    
## =======================================================================================    
## Data loader
## ======================================================================================= 
## This colormap values is your labels in your GT 
colormap = [[255,0,0],[79,255,130],[198,118,255],[255,225,10],[84,226,255],[255,121,193]]
num_classes = len(colormap)
cm2lbl = np.zeros(256**3) # Every pixel is in range of 0 ~ 255，RGB with 3 channels

for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 
def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64') # 

## This is where the example data located
#==============================================================================
# simply use the code below to write the txt files
# for i in range(0,523):
#     append_text = f'trainA/{i}.png trainB/{i}.png \n'
#     my_file = open('D:\\fuelcell_SR\\CNN_superresolved\\trainnopore.txt', 'a')
#     my_file.write(append_text)
#     my_file.close()
#=============================================================================
ROOT = "F:\\python_Code\\Segmentation_code\\Example\\"
# Reading image data
def read_image(mode="train", val=False):
    if(mode=="train"):    # 加载训练数据
        filename = ROOT + "\\train.txt"
    elif(mode == "test"):    # 加载测试数据
        filename = ROOT + "\\test.txt"

    data = []
    label = []
    with open(filename, "r") as f:
        images = f.read().split()
        for i in range(len(images)):
            if(i%2 == 0):
                data.append(ROOT+images[i])
            else:
                label.append(ROOT+images[i])
               
    print(mode+":contains: "+str(len(data))+" images")
    print(mode+":contains: "+str(len(label))+" labels")
    return data, label

data, label = read_image("train")
   

def crop(data, label, height=patch_size, width=patch_size):
    st_x = 0
    st_y = 0
    box = (st_x, st_y, st_x+width, st_y+height)
    data = data.crop(box)
    label = label.crop(box)
    return data, label

def image_transforms(data, label, height=patch_size, width=patch_size):
    data, label = crop(data, label, height, width)
    # convert to tensor, and normalization
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        # tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    data = im_tfs(data)
    label = np.array(label)
    label = image2label(label)
    label = torch.from_numpy(label).long()   # CrossEntropyLoss require a long() type
    return data, label

class SegmentDataset(torch.utils.data.Dataset):
    
    # make functions
    def __init__(self, mode="train", height=patch_size, width=patch_size, transforms=image_transforms):
        self.height = height
        self.width = width
        self.transforms = transforms
        data_list, label_list = read_image(mode=mode)
        self.data_list = data_list
        self.label_list = label_list
        
    
    # do literation
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        img= img.convert('RGB')
        lb = Image.open(label)
 
        label= lb.convert('RGB')
        img, label = self.transforms(img, label, self.height, self.width)
        return img, label
    
    def __len__(self):
        return len(self.data_list)

height = patch_size
width = patch_size
Segment_train = SegmentDataset(mode="train")
# Segment_valid = SegmentDataset(mode="val")
Segment_test = SegmentDataset(mode="test")

train_data = DataLoader(Segment_train, batch_size=batchsize, shuffle=True)
# valid_data = DataLoader(Segment_valid, batch_size=8)
test_data = DataLoader(Segment_test, batch_size=batchsize)
## =======================================================================================    
## Data loader end
## ======================================================================================= 

## optimizer and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_decay, 
                                                       patience=5, verbose=False, threshold=0.0001, 
                                                       threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
criterion = nn.CrossEntropyLoss()

# train data record
train_loss = []
train_acc = []
train_acc_cls = []
train_mean_iu = []

# validation data record
test_loss = []
test_acc = []
test_acc_cls = []
test_mean_iu = []
test_fwavacc = []

for epoch in range(EPOCH):
    _train_loss = 0
    _train_acc = 0
    _train_acc_cls = 0
    _train_mean_iu = 0
    _train_fwavacc = 0
    _each_acc_train = 0
    prev_time = datetime.now()
    ##=========================================================
    # Train model
    ##=========================================================
    net = net.train()
    for step, (x,label) in enumerate(train_data):
        if torch.cuda.is_available():
            x = x.cuda()
            label = label.cuda()

            
        out = net(x) 
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _train_loss += loss.item()

        label_pred = out.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        for lbt, lbp in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
            _train_acc += acc
            _train_acc_cls += acc_cls
            _train_mean_iu += mean_iu
            _train_fwavacc += fwavacc
        
    # recold loss and acc in the epoch
    train_loss.append(_train_loss/len(train_data))
    train_acc.append(_train_acc/len(Segment_train))
    train_mean_iu.append(_train_mean_iu/len(Segment_train))
    
    epoch_str = ('Epoch: {}, train Loss: {:.5f}, train Weight Acc: {:.5f}, train UNWeight Acc: {:.5f} '.format(
        epoch, _train_loss / len(train_data), _train_acc / len(Segment_train), _train_mean_iu / len(Segment_train)))
    print(epoch_str)
    
    ##=========================================================
    # validation
    ##=========================================================
    net = net.eval()
    _test_loss = 0
    _test_acc = 0
    _test_acc_cls = 0
    _test_mean_iu = 0
    _test_fwavacc = 0
    _test_ave_acc = 0

    for img_data, img_label in test_data:
        if torch.cuda.is_available():
            im = img_data.cuda()
            label = img_label.cuda()
        else:
            im = img_data
            label =img_label
        
        # forward
        out = net(im)
        loss = criterion(out, label)
        _test_loss += loss.item()
        label_predtest = out.max(dim=1)[1].data.cpu().numpy()
        label_truetest = label.data.cpu().numpy()
        for lbt, lbp in zip(label_truetest, label_predtest):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
            _test_acc += acc
            _test_acc_cls += acc_cls
            _test_mean_iu += mean_iu

    test_loss.append(_test_loss/len(test_data))
    test_acc.append(_test_acc/len(Segment_test))
    test_mean_iu.append(_test_mean_iu/len(Segment_test))
    ## Schedule learning rate
    scheduler.step(_test_loss/len(test_data))
    
    epoch_str = ('Epoch: {}, Test Loss: {:.5f}, Test Weight Acc: {:.5f}, Test UNWeight Acc: {:.5f} '.format(
            epoch, _test_loss / len(test_data), _test_acc / len(Segment_test), _test_mean_iu / len(Segment_test)))
    print(epoch_str)
    print('Epoch:', epoch, '| Learning rate_D', optimizer.state_dict()['param_groups'][0]['lr'])
    print('')
    
    ##=========================================================
    # Save model
    ##=========================================================
    if epoch == start_epoch_forsave:
        PATH = 'F:\\python_Code\\Segmentation_code\\Example\\checkpoint\\%d' % (epoch) +'.pt'
    
        torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)
        start_epoch_forsave += save_interval
            
# train loss visualization  
plt.figure()  
epoch = np.array(range(EPOCH))
plt.plot(epoch, train_loss, label="train_loss")
plt.plot(epoch, test_loss, label="validation_loss")
plt.title("loss during training")
plt.legend()
plt.grid()
plt.show()

plt.figure()  
# train acc/ valid acc visualization    
plt.plot(epoch, train_acc, label="train_acc")
plt.plot(epoch, train_mean_iu, label="train_mIOU")
plt.plot(epoch, test_acc, label="validation_acc")
plt.plot(epoch, test_mean_iu, label="validation_mIOU")
plt.title("accuracy during training")
plt.legend()
plt.grid()
plt.show()

#visualization
import random as rand
def predict(img, label): # prediction
    img = img.unsqueeze(0).cuda()
    out = net(img)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    return pred, label

def show(size=256, num_image=4, img_size=10, offset=0, shuffle=False):
    _, figs = plt.subplots(num_image, 3, figsize=(img_size, img_size))
    for i in range(num_image):
        if(shuffle==True):
            offset = rand.randint(0, min(len(Segment_train)-i-1, len(Segment_test)-i-1))
        img_data, img_label = Segment_test[i+offset]
        pred, label = predict(img_data, img_label)
        min_val,max_val,min_indx,max_indx = cv2.minMaxLoc(pred) 
        print('pred: ', min_val,max_val,min_indx,max_indx)
        img_data = Image.open(Segment_test.data_list[i+offset])
        img_label = Image.open(Segment_test.label_list[i+offset])
        img_data, img_label = crop(img_data, img_label) 
        img_label = img_label.convert('RGB')
        img_label = image2label(img_label)
        min_val,max_val,min_indx,max_indx = cv2.minMaxLoc(img_label) 
        print('Label: ',min_val,max_val,min_indx,max_indx)
        figs[i, 0].imshow(img_data)  
        figs[i, 0].axes.get_xaxis().set_visible(False)  
        figs[i, 0].axes.get_yaxis().set_visible(False)  
        figs[i, 1].imshow(img_label)                  
        figs[i, 1].axes.get_xaxis().set_visible(False) 
        figs[i, 1].axes.get_yaxis().set_visible(False)  
        figs[i, 2].imshow(pred)                       
        figs[i, 2].axes.get_xaxis().set_visible(False)  
        figs[i, 2].axes.get_yaxis().set_visible(False)  

    # titles
    figs[num_image-1, 0].set_title("Image", y=-0.2*(10/img_size))
    figs[num_image-1, 1].set_title("Label", y=-0.2*(10/img_size))
    figs[num_image-1, 2].set_title("U-resnet", y=-0.2*(10/img_size))
    plt.show()

show(offset=30)

