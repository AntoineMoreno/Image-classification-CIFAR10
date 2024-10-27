# app.py
from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
import numpy as np

app = Flask(__name__)

labels_to_names = {
    0 : "airplane",
    1 : "automobile",
    2 : "bird", 
    3 : "cat", 
    4 : "deer", 
    5 : "dog", 
    6 : "frog", 
    7 : "horse", 
    8 : "ship", 
    9 : "truck"
}

logging.basicConfig(level=logging.INFO)

# Define the model architecture (same as used during training)
class NaturalSceneClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(4096,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    def forward(self, xb):
        return self.network(xb)

# Load the model
model = NaturalSceneClassification()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])


#Set the default page to be the one from intemplates/index.html
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Redirect to predict route
        return redirect(url_for('predict'))
    return render_template('index.html')

#Handle the image uploading
@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info('Received prediction request')

        # Check if an image file was uploaded
        if 'file' not in request.files:
            error_message = 'No file part in the request'
            return render_template('index.html', prediction=error_message), 400

        file = request.files['file']
        if file.filename == '':
            error_message = 'No file selected'
            return render_template('index.html', prediction=error_message), 400

        # Read the image and prepare it to be fed into the model
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)

        # Make prediction
        with torch.no_grad():
            output = model(batch_t)
            _, predicted = torch.max(output.data, 1)
            predicted_class = labels_to_names[predicted[0].item()]

        # Return the result on the web page
        return render_template('index.html', prediction=predicted_class), 200

    except Exception as e:
        logging.exception("Error during prediction")
        error_message = f"Error: {str(e)}"
        return render_template('index.html', prediction=error_message), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)