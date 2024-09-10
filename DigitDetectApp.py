from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np


canvas_width = 300
canvas_height = 300
brush_size = 5
brush_color = 255
img = Image.new("L", (canvas_width, canvas_height), color="black")  # image created with this height and width and used as canvas L is for grayscale
draw_img = ImageDraw.Draw(img)
value = None


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
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()  # Set the model to evaluation mode


def paint(event):
    """Draw on both the tkinter canvas and the Pillow image when dragging the mouse."""
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    
    # Draw on tkinter canvas (just visual reference)
    canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
    
    # Draw on the Pillow image for saving later
    draw_img.ellipse([x1, y1, x2, y2], fill=brush_color, outline=brush_color)


def make_tensor():
    """Make Tensor of current image."""
    # Resize the image to 28x28 pixels
    img_resized = img.resize((28, 28))
    
    trans_to_tensor = ToTensor()
    return trans_to_tensor(img_resized)

def predict_digit():
    """Predict digit from current image."""
    data = make_tensor()
    data = data.unsqueeze(0).to(device) #add batch dimension (how many samples are passed at once used for parallel processing)
    output = model(data)
    prediction = output.argmax(dim=1, keepdim=True).item()  #it is a tensor of shape 1, 10 as our batch size was one (one digit was being detected)
                                                            #dim=1 means find the max value of each column (go through the rows) 0 is row dimension 1 column
                                                            #keepdim keeps original shape of tensor so if [[x,r,t]] was  put in output is [[max]] item just gets the single value (dangerous)

    predict_value.config(text=f"Detected Digit: {prediction}  ", font=("Arial", 18))
    predict_value.update_idletasks()
    print("prediction {}".format(prediction))

def clear():
    canvas.delete("all")
    global img, draw_img
    img = Image.new("L", (canvas_width, canvas_height), color="black")
    draw_img = ImageDraw.Draw(img)
    

# Create the window
window = tk.Tk()
window.title("Digit Recognition CNN App")


frame = tk.Frame(window)
frame.pack()


canvas = tk.Canvas(frame, width=canvas_width, height=canvas_height, bg="black")
canvas.grid(row=0, column=0)  


predict_value = tk.Label(frame, text=f"PRESS DETECT!", font=("Arial", 14))
predict_value.grid(row=0, column=1, padx=20) 

# Binds mouse drag event to the paint function
canvas.bind("<B1-Motion>", paint)

detect_digit = tk.Button(window, text="DETECT", command=predict_digit)
detect_digit.pack()

clear_button = tk.Button(window, text="Clear", command=clear)
clear_button.pack()

# Run the event loop
window.mainloop()

