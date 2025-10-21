# Amharic Sign Language → Text
**Amharic sign language to text translation using image processing and machine learning**

**Short description:** A computer-vision system to recognize Ethiopian Sign Language (ESL) alphabets and translate to text using a fine-tuned SSD MobileNet v2 FPN Lite model.

## Abstract
In this repository, we will try to address the communication challenges faced by the 1–2.5 million hearing-impaired individuals in Ethiopia by proposing an Ethiopian Sign Language (ESL) recognition system using computer vision and deep learning techniques. The objective is to bridge the communication gap between the hearing-impaired community and the general population, fostering social inclusion and collaboration. To achieve this, collected a custom dataset of Ethiopian Sign Language alphabets using a webcam and manually labeled the data. Then fine-tuned a pre-trained **SSD MobileNet v2 FPN Lite 320×320** object detection model by optimizing hyperparameters and leveraging GPU acceleration during training. The performance of the model was evaluated using precision, recall, and loss metrics. Future improvements include expanding the vocabulary, collecting more diverse data, increasing training steps, and optimizing the model for mobile devices using TensorFlow Lite.

## Demo
<img width="667" height="527" alt="image" src="https://github.com/user-attachments/assets/189feacc-336e-4a4d-b460-a303ac28bc9a" />


## Repo contents
- `train.py` — training script in python code using Jupyter notebook.
- `eval.py` — evaluation script in python code using Jupyter notebook.
- `detect_demo.py` — demo script in python code using Jupyter notebook.
- 
## Quick start
1. Create a Python virtual environment (Python 3.9+).
2. Install dependencies for TensorFlow and Cuda

