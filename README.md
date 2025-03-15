# Satellite Image Segmentation using U-Net

## Project Overview
This project focuses on satellite image segmentation using a U-Net deep learning model. The main goal is to efficiently classify and segment different regions in satellite images. The model is trained, tested, and deployed on Hugging Face for real-time predictions.

🔗 Try the Model Here: [Hugging Face Interface](https://huggingface.co/spaces/Salll/satellite_image_segmentation_prediction)

## Project Workflow
The project follows a structured approach:

**1. Pre-processing the Satellite Image Dataset**
Loaded and pre-processed the dataset for image segmentation.
Applied normalization, resizing, and augmentation techniques to improve model performance.

**2. Training the U-Net Model**
Implemented a U-Net architecture for segmentation tasks.
Used appropriate loss functions and optimizers to improve accuracy.

**3. Local Debugging & Testing**
Performed error analysis and fine-tuned the model for better performance.
Debugged issues with training and validation losses.

**4. Prediction & Performance Evaluation**
Ran the trained model on test images to check segmentation accuracy.
Generated and analyzed segmentation masks.

**5. Deployment on Hugging Face**
Exported the model and deployed it on Hugging Face Spaces.
Created an interactive interface for users to upload satellite images and get segmented outputs.

## How to Use the Model?
Visit the Hugging Face link provided above.

Upload a satellite image to the interface.

Click Predict to get the segmented output.

The model will generate a segmentation mask, highlighting different regions.

## Technologies Used
**Deep Learning**: U-Net Model

**Libraries**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib

**Framework**: Jupyter Notebook

**Deployment**: Hugging Face Spaces

## Installation & Setup
1. Clone the Repository
bash
Copy code
git clone <repo-link>
cd <repo-folder>
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. Run the Jupyter Notebook
bash
Copy code
jupyter notebook
Load and execute Satellite_imagery(Prediction with custom images+activation output).ipynb.
## Project Structure
bash

Copy code

📂 Satellite-Image-Segmentation

│── 📂 data/                 # Dataset directory

│── 📂 models/               # Trained U-Net model

│── 📂 notebooks/            # Jupyter Notebooks

│── 📂 scripts/              # Python scripts for processing

│── 📜 requirements.txt      # Dependencies

│── 📜 README.md             # Project documentation

## Future Improvements
Enhance model performance by training on larger datasets.
Optimize deployment for faster inference speeds.
Implement multi-class segmentation for detecting more land types.
