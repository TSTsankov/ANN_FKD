# import the usual resources
import matplotlib.pyplot as plt
import numpy as np

# import utilities to keep workspaces alive during model training
#from workspace_utils import active_session

# watch for any changes in model.py, if it changes, re-load it automatically
#%load_ext autoreload
#%autoreload 2

## TODO: Define the Net in models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
from models import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.version)
net = Net().to(device)
print(net)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor

## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop

# testing that you've defined a transform

data_transform = transforms.Compose(
    [Rescale(250), RandomCrop(224),
     Normalize(), ToTensor()])
assert (data_transform is not None), 'Define a data_transform'

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(
    csv_file=
    "D:\\Users\\Tsvetan\\FootDataset\\person_keypoints_train2017_foot_v1\\NEWOUT.csv",
    root_dir=
    "D:\\Users\\Tsvetan\\FootDataset\\person_keypoints_train2017_foot_v1\\out\\",
    transform=data_transform)

print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())

# load training data in batches
batch_size = 32

train_loader = DataLoader(
    transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# load in the test data, using the dataset class
# AND apply the data_transform you defined above

# create the test dataset
test_dataset = FacialKeypointsDataset(
    csv_file=
    "D:\\Users\\Tsvetan\\FootDataset\\person_keypoints_val2017_foot_v1\\NEWOUT.csv",
    root_dir=
    "D:\\Users\\Tsvetan\\FootDataset\\person_keypoints_val2017_foot_v1\\ready_val_out\\",
    transform=data_transform)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# test the model on a batch of test images


def net_sample_output():

    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.cuda.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)

        # reshape to batch_size x 68 x 2 pts ##CHANGED
        output_pts = output_pts.view(output_pts.size()[0], 6, -1)

        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts


test_images, test_outputs, gt_pts = net_sample_output()

# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(
        predicted_key_pts[:, 0],
        predicted_key_pts[:, 1],
        s=20,
        marker='.',
        c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=4):
    fig = plt.figure()
    iimg = 1
    sp = 1
    for i in range(batch_size):

        iimg += 1
        if i % 2 == 0:
            sp += 1
            iimg = 1
        fig.add_subplot(sp, batch_size, iimg)

        # un-transform the image data
        image = test_images.cpu(
        )[i].data  # get the image from it's Variable wrapper
        image = image.numpy()  # convert to numpy array from a Tensor
        image = np.transpose(
            image, (1, 2, 0))  # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs.cpu()[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts * 50.0 + 100

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100

        # call show_all_keypoints
        show_all_keypoints(
            np.squeeze(image), predicted_key_pts, ground_truth_pts)

        plt.axis('off')

    plt.show()


# call it
#visualize_output(test_images, test_outputs, gt_pts)

## TODO: Define the loss and optimization
import torch.optim as optim

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(params=net.parameters(), lr=0.001)


def train_net(n_epochs):
    print("Training...")

    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)
            #output_pts = output_pts.type(torch.cuda.FloatTensor)
            #print(output_pts.type)
            #print(key_pts.type)
            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 32 == 31:  # print every 10 batches
                print(datetime.datetime.now())
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(
                    epoch + 1, batch_i + 1, running_loss / 1000))
                running_loss = 0.0

    print('Finished Training')


# train your network
n_epochs = 500  # start small, and increase when you've decided on your model structure and hyperparams

# this is a Workspaces-specific context manager to keep the connection
# alive while training your model, not part of pytorch
#with active_session():

#train_net(n_epochs)

## TODO: change the name to something uniqe for each new model
model_dir = 'D:\\Users\\Tsvetan\\FootDataset\\'
model_name = 'keypoints_model_garima.pt'

cpnt = torch.load(model_dir + "500" + model_name)
print()
print()
for key in cpnt.keys():
    print(key + " {}".format(cpnt[key]))
print()
print()
net.load_state_dict(torch.load(model_dir + "500" + model_name))

train_net(10)

net.eval()

# after training, save your model parameters in the dir 'saved_models'
#torch.save(net.state_dict(), model_dir + model_name)

# Get the weights in the first conv layer, "conv1"
# if necessary, change this to reflect the name of your first conv layer
weights1 = net.conv1.weight.data.cpu()

w = weights1.numpy()

filter_index = 0

print(w[filter_index][0])
print(w[filter_index][0].shape)

# display the filter weights
#plt.imshow(w[filter_index][0], cmap='gray')

####
test_images, test_outputs, gt_pts = net_sample_output()
visualize_output(test_images, test_outputs, gt_pts)
