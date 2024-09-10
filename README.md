# Digit-Detection-CNN
Convolutional neural network used to detect hand written digits (0-9)

# Set up
To run create a python virtual enviornment with all dependacies:
1. in project folder run the command pip install virtualenv (to install venv) and then
2. python -m venv <virtual-environment-name> 
3. go to scripts folder in <virtual-environment-name>  folder open bash window and run the command "start activate.bat" should open command terminal with your venv running (can also do activate.bat in terminal)
4. in terminal go back out project folder level and run the command pip install -r requirements.txt
5. open vscode and do ctrl + shift + p select your virtual enviornment

# Run Application
1. run DigitDetectApp.py
2. use mouse on black canvas to draw a digit and press detect for a digit detection 0-9
3. press clear to clean canvas to write another digit

# Train your own model
(a little advanced)
1. In terminal run the command jupyter notebook
2. In jypter notebook click new button and select python 3 (ipykernel)
3. paste the "TrainNetwork.py" into a jupyter notebook cell 
4. ajdust learning rate and epochs (set to 10 for now) for training, Optimizer(learning algorithim) being used: Adam
5. Hit run
6. console will contain epoch number loss and model accuracy (smaller loss numbers are from the testing as the loss is divided by test batch size)
7. Add this line into a cell "torch.save(model.state_dict(), "modelName.pth")"
8. add "modelName.pth" to project folder and rename to "mnist_cnn.pth"
9. do steps in "Run Application"

# Demo Vid!
https://youtu.be/lZAf13rlaac
