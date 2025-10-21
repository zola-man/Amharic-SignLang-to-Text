# Amharic Sign Language → Text
**Amharic sign language to text translation using image processing and machine learning**

**Short description:** A computer-vision system to recognize Ethiopian Sign Language (ESL) alphabets and translate to text using a fine-tuned SSD MobileNet v2 FPN Lite model.

## Abstract
In this paper, we address the communication challenges faced by the 1–2.5 million hearing-impaired individuals in Ethiopia by proposing an Ethiopian Sign Language (ESL) recognition system using computer vision and deep learning techniques. Our objective is to bridge the communication gap between the hearing-impaired community and the general population, fostering social inclusion and collaboration. To achieve this, we collected a custom dataset of Ethiopian Sign Language alphabets using a webcam and manually labeled the data. We then fine-tuned a pre-trained **SSD MobileNet v2 FPN Lite 320×320** object detection model by optimizing hyperparameters and leveraging GPU acceleration during training. The performance of the model was evaluated using precision, recall, and loss metrics. Future improvements include expanding the vocabulary, collecting more diverse data, increasing training steps, and optimizing the model for mobile devices using TensorFlow Lite.

## Demo
![demo gif](assets/demo.gif)

## Repo contents
- `notebooks/01-esl-experiment.ipynb` — Jupyter notebook with preprocessing, training notes, and evaluation.


## Quick start
1. Create a Python virtual environment (Python 3.9+).
2. Install dependencies:
```bash
pip install -r requirements.txt
