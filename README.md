# Malaysia Car Plate Detection

This project is a **car plate detection and recognition system** built using **YOLO11n** for object detection and **EasyOCR** for reading the text on the plates. The entire system is deployed on **Gradio**, allowing you to easily test and interact with it through a user-friendly interface.

## Features
- Detects car plates in images.
- Recognizes the characters on the detected plates.
- Interactive interface using Gradio.

## How to Run
1. Open `carplate_detection.ipynb`.
2. Run the notebook to train or fine-tune the model. All dataset processing and model fine-tuning are included in the notebook.

3. Run index.py to get the interface in gradio (pip install -r requirements.txt)
4. Make sure the pt is saved in the folder detector and named as car_plate_detection_yolov11n
## Datasets
- **Original dataset:** [Link](https://universe.roboflow.com/gocar/malaysia-car-plate-number/dataset/4)  
- **Dataset with cars at night:** [Link](https://drive.google.com/drive/folders/1fLB_Ui03iKvTCJENPOCUrFyxGEUKzEqy?usp=drive_link)  
- **Testing dataset (for trying the system):** [Link](https://drive.google.com/drive/folders/1fLB_Ui03iKvTCJENPOCUrFyxGEUKzEqy?usp=drive_link)
## Model
- **Model:** [Link](https://drive.google.com/file/d/1BoLW52WVJEZUvFtM2YwCx82zYKhybopL/view?usp=drive_link)
