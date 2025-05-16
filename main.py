import os
import torch
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
from architecture import ResNetLungCancer  # Ensure this is the correct import for your model

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNetLungCancer(num_classes=4)
model.load_state_dict(torch.load('Model/lung_cancer_detection_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names for the model output
class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

# Storage for previous results
previous_results = []

@app.route('/')
def index():
    # Calculate cancer type percentages
    total_cases = len(previous_results)
    type_counts = {t: 0 for t in class_names}
    
    for result in previous_results:
        type_counts[result['cancer_type']] += 1
    
    type_percentages = {t: (count / total_cases * 100) if total_cases > 0 else 0 for t, count in type_counts.items()}

    return render_template('index.html', previous_results=previous_results[-5:], type_percentages=type_percentages)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'scanUpload' not in request.files:
            return redirect(request.url)
        file = request.files['scanUpload']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            result, cancer_status, suggestion = predict(file_path)

            # Store the detection result
            previous_results.append({
                "filename": filename,
                "cancer_type": result,
                "status": cancer_status
            })

            return render_template('result.html', filename=filename, result=result, cancer_status=cancer_status, suggestion=suggestion)
    return render_template('detect.html')

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    predicted_class = torch.argmax(output, dim=1).item()
    
    # Assuming the model outputs probabilities for each class
    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    # Define a threshold for cancer detection
    cancer_threshold = 0.5  # Adjust this threshold based on your model's output
    cancer_detected = probabilities[1] + probabilities[3]  # Assuming class 1 and 3 are cancer classes

    if cancer_detected > cancer_threshold:
        cancer_status = "Cancer Detected"
        suggestion = "Please consult a healthcare professional immediately."
    else:
        cancer_status = "No Cancer Detected"
        suggestion = "You are advised to maintain regular health check-ups."

    return class_names[predicted_class], cancer_status, suggestion

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True)
