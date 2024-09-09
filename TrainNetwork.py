from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from PIL import Image

train_data = datasets.MNIST(
    root = 'data',
    train = True,               #if data is to be used to train or not
    transform = ToTensor(),     #creating an instance of ToTensor and assigning it to transform to be used for transforming images
    download = True
)

test_data = datasets.MNIST(     #array is 60k by 28 and 28 which is size of image 28 pixels
    root = 'data',
    train = False,
    transform = ToTensor(),
    download = True
)

loaders = {                     #load in data in batches
    'train' : DataLoader(train_data, batch_size=100, shuffle= True, num_workers = 1),
    'test' : DataLoader(test_data, batch_size=100, shuffle= True, num_workers = 1) 
}

class CNN(nn.Module):

    def __init__(self) -> None:
        super(CNN, self).__init__()  #call init of the parent using current class and this instance can be written as super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)  # input channels is just how many types of input are given only 1 type no rgb used so only gray scale is to be worried about, output channel filters are just the 10 different types features (patterns) that are attempting to be learned kernel is just filter size 5x5 grids will be used to gauge these features (slides across image)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d() #used to deactivate and drop certain nodes using some probability
        self.fc1 = nn.Linear(320, 50) #defines a fully connected layer what you see in diagrams 320 rows 50 columns each column has bias with it 50 nodes with 320 connections to it
        self.fc2 = nn.Linear(50, 10) # 10 is your last with probabilities


    def forward(self, x):   #automatically called nn.Module
        x = F.relu(F.max_pool2d(self.conv1(x), 2))    #relu! f(x) = max(0, x) # changes size of filter map to 2x2 to hold more complex patterns
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) #builds on previous 
        x = x.view(-1, 320) #-1 for inferring batch size #320 makes sense cause 2*2 feature map 4 parts of it 2*2 again to make 4 parts of 4 4*4 = 16 for the 20 channels 16 * 20 = 320
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x) # softmax so values are put inbetween 0 and 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #CUDA !!!!! nvidia gpu check
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001) #learning rate is 0.001

loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_index, (data, target) in enumerate(loaders['train']):# data is 28 x 28 target is 0-9 loading in 100 at a time
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()   #zero the gradient before forward pass and backpropogation to avoid accumulation of gradients so new weights and biases are not skewed by a previous batch the values that are set
                                #to it are what start as an inital value to be modified by optimizer.step()
        output = model(data)    #forward pass passing in 100 images (data)
        loss = loss_fn(output, target)
        loss.backward()     #backPropogation
        optimizer.step()    #update weights
        if batch_index % 20 == 0:
            print("epoch {} Total processed: {}/{}".format(epoch, batch_index * len(data), len(loaders['train'])))
            print("loss: {:.2f}".format(loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item() #for 100 intstances check if target and prediction are the same if prediciton classification aligns with target

    test_loss /= len(loaders['test'].dataset) #this is the 10000 used to test
    print("loss = {:.2f} Accuracy = {:.2f}".format(test_loss, (correct/len(loaders['test'].dataset))))


#training the moodel and testing it for 10 epochs
for epoch in range(1, 11):  #10 epochs
    train(epoch)
    test()


#torch.save(model.state_dict(), "mnist_cnn.pth") #save the model to a file

#print(train_data.data.shape)    #shape of tensor
#print(train_data.targets.size()) # num of images data set
#print(train_data.targets) # gives u the target the data set is trying to reach 0-9

# code to manually test the model

'''
model.eval()

data, target = test_data[0]

data = data.unsqueeze(0).to(device) #add batch dimension (how many samples are passed at once used for parallel processing)
output = model(data)

prediction = output.argmax(dim=1, keepdim=True).item()

print("prediction {}".format(prediction))

image = data.squeeze(0).squeeze(0).cpu().numpy()

plt.imshow(image, cmap="gray")
plt.show()


----------------------------------------------------------------------------------------------------------------------------

model.eval() #sets model to evaluation mode as model contains drop out layers and other layers that behave differently in training and testing

path = r"useurown.jpg"
img = Image.open(path)

trans_to_tensor = ToTensor()
data = trans_to_tensor(img)

data = data.unsqueeze(0).to(device) #add batch dimension (how many samples are passed at once used for parallel processing)
output = model(data)

print(output.shape)
print(output.argmax(dim=1, keepdim=True))
prediction = output.argmax(dim=1, keepdim=True).item()  #it is a tensor of shape 1, 10 as our batch size was one (one digit was being detected)
                                                        #dim=1 means find the max value of each column (go through the rows) 0 is row dimension 1 column
                                                        #keepdim keeps original shape of tensor so if [[x,r,t]] was  put in output is [[max]] item just gets the single value (dangerous)

print("prediction {}".format(prediction))

image = data.squeeze(0).squeeze(0).cpu().numpy()

plt.imshow(image, cmap="gray")
plt.show()

#print(img_tensor.shape)
#data, target = test_data[0]
#print(data.shape)
#data, target = test_data[0]

#print(type(test_data[0]))

'''