from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import io
import pickle
import os

app = Flask(__name__)
CORS(app)

# Function to load the model based on the filename
def load_custom_model(model_filename):
    model_path = os.path.join("models", model_filename + ".h5")
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

# # Load the ResultMap for mapping predictions to face names
# with open("ResultsMap.pkl", 'rb') as fileReadStream:
#     ResultMap = pickle.load(fileReadStream)

# Function to predict a face from an image
def predict_face(image_data, model, filename):
    map_path = os.path.join("models", filename + "_map.pkl")
    # Load the ResultMap for mapping predictions to face names
    with open(map_path, 'rb') as fileReadStream:
        ResultMap = pickle.load(fileReadStream)

    img = image.load_img(io.BytesIO(image_data), target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array, verbose=0)
    # Get the predicted class index
    predicted_class_index = np.argmax(result)
    # Get the predicted face name
    predicted_face_name = ResultMap[predicted_class_index]
    # Get the confidence score
    confidence_score = result[0][predicted_class_index] * 100  # Convert to percentage

    top_indices = np.argsort(result)[0][-5:][::-1]

    for i in range(0, 5):
        top_indices[i] = ResultMap[top_indices[i]]


    with open("results.txt", 'a') as file:
            # Write the text to the file
            file.write(filename + "\t" + str(top_indices) + '\n')
    print(result)

    return predicted_face_name, confidence_score

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'filename' not in request.form:
        return jsonify({'error': 'Missing image or filename part'})
    
    image_data = request.files['image'].read()
    filename = request.form['filename']
    
    if not image_data or not filename:
        return jsonify({'error': 'Missing image data or filename'})

    # Load the model based on the filename
    model = load_custom_model(filename)
    if model is None:
        return jsonify({'error': 'Model not found'})

    predicted_face, confidence_score = predict_face(image_data, model,filename)
    return jsonify({'predicted_face': predicted_face, 'confidence_score': confidence_score})

# Route to return a list of .h5 files in the models directory
@app.route('/getAvailableClasses', methods=['GET'])
def list_h5_files():
    # Get the models directory
    models_directory = os.path.join(os.getcwd(), "models")
    # List all files in the directory
    all_files = os.listdir(models_directory)
    # Filter .h5 files
    h5_files = [filename[:-3] for filename in all_files if filename.endswith('.h5')]
    return jsonify({'h5_files': h5_files})

if __name__ == '__main__':
    app.run(debug=True, port=4000)
    print("Flask server is running...")
