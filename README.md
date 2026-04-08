# HematoVision - Blood Cell Classification using Deep Learning and Flask

## Objective
HematoVision is a Flask web application that classifies blood cell microscope images into four classes.
It uses a trained CNN model for inference from uploaded images.

## Dataset
- Source: Kaggle Blood Cell Dataset
- Classes: Eosinophil, Lymphocyte, Monocyte, Neutrophil
- Dataset files are not included in this repository.

## Model
- Custom CNN for 4-class classification
- Input image size: 224x224
- Normalization: `rescale = 1./255`
- Inference model: `HematoVision/Blood_Cell.h5`

## Accuracy
- Test accuracy: ~80%

## How To Run
1. Clone this repository.
2. Go to the project folder:
```bash
cd HematoVision
```
3. Create and activate a virtual environment.
4. Install dependencies:
```bash
pip install -r requirements.txt
```
5. Run the Flask app:
```bash
python app.py
```
6. Open `http://127.0.0.1:5000` and upload an image.

## Demo
![Home Page](HematoVision/screenshots/home_page.png)

![Prediction Result](HematoVision/screenshots/result_page.png)