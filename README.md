<h2>CNN Image Classification Web Application</h2>
Project Description

This project is a web-based image classification application developed using a Convolutional Neural Network (CNN) and the CIFAR-10 dataset.

The application uses Flask to provide a web interface where users can upload an image and receive a prediction from the trained CNN model. The model classifies images into 10 categories:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck

The CNN model achieved approximately 82% accuracy on the CIFAR-10 test dataset. The project demonstrates how a trained machine learning model can be integrated into a web application for real-time image classification.

<h2>About the Model</h2>

The model was trained using the CIFAR-10 dataset, which contains 60,000 colour images across 10 different classes. Each image has a resolution of 32 × 32 pixels.

Before an image is passed to the model, it is:

Converted to RGB format.
Resized to 32 × 32 pixels.
Converted into a NumPy array.
Normalized by dividing pixel values by 255.
Passed to the trained CNN model for prediction.

The model achieved approximately 82% test accuracy.

However, the model has some limitations. Since CIFAR-10 images are very small, some visual details can be lost. The model may also confuse visually similar classes, such as dogs, deer, and horses. Images uploaded from outside the CIFAR-10 dataset may also produce incorrect predictions because of differences in image quality, background, lighting, and object appearance.

The trained model is stored in:

model/cnn_cifar10_model.keras
<h2>How to Run the Project</h2>
1. Install Python

This project uses:

Python 3.11.9

Check your Python version:

python --version

If multiple Python versions are installed on Windows, you can check Python 3.11 using:

py -3.11 --version
2. Clone the Repository

Clone the GitHub repository:

git clone https://github.com/1036025-hub/cnn_project_deployment.git

Move into the project folder:

cd cnn_project_deployment
3. Create a Virtual Environment

Create a Python 3.11 virtual environment:

py -3.11 -m venv venv
4. Activate the Virtual Environment

On Windows:

venv\Scripts\activate

After activation, you should see:

(venv)

at the beginning of your terminal.

Check the Python version:

python --version

It should show:

Python 3.11.9
5. Install the Required Packages

Upgrade pip:

python -m pip install --upgrade pip

Install the project dependencies:

pip install -r requirements.txt
6. Run the Flask Application

<h2>Start the application:</h2>

python app.py

The terminal should show that the CNN model has loaded successfully.

Open a web browser and go to:

http://127.0.0.1:5000

You can then upload an image and click Predict Image to see the model's prediction.

<h2>Future Improvements</h2>

There are several ways this project could be improved in the future.

<h2>Model Improvements</h2>
Increase the CNN model depth and complexity.
Use additional data augmentation techniques.
Improve hyperparameter tuning.
Experiment with different CNN architectures.
Use transfer learning with a pretrained model.
Train the model for longer with optimized parameters.
Improve classification between visually similar classes such as dogs, deer, and horses.
Web Application Improvements
Add drag-and-drop image uploading.
Display the top 3 predictions instead of only the highest prediction.
Add prediction history.
Add confidence charts.
Improve the user interface and mobile responsiveness.
Add better image validation and upload security.
Add a loading indicator while the model is making a prediction.
Improve error handling for unsupported or invalid images.
<h2>Technologies Used</h2>
Python 3.11.9
TensorFlow
Keras
NumPy
Pillow
Flask
HTML
CSS
Bootstrap
CIFAR-10 Dataset
