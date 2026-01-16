# 🖼️ Automatic Image Captioning (CNN + LSTM)

An advanced Deep Learning project that generates natural language descriptions for images. This system bridges Computer Vision and Natural Language Processing (NLP) by combining Convolutional Neural Networks (CNN) for feature extraction with Recurrent Neural Networks (LSTM) for text generation.

## 🧠 Model Architecture
The project implements a hybrid **Encoder-Decoder** architecture:
- **Encoder (CNN):** Uses a pre-trained **VGG16 / ResNet50** model (transfer learning) to extract high-level visual features from images. The final classification layer is removed to obtain raw feature vectors.
- **Decoder (RNN):** A **Long Short-Term Memory (LSTM)** network takes the image feature vectors as input and generates captions word-by-word, learning the sequential dependency of language.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras
- **Language:** Python 3.x
- **Computer Vision:** OpenCV, PIL
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib

## 📊 Dataset & Performance
- **Dataset:** Trained on the **Flickr8k dataset**, consisting of 8,000 images, each paired with 5 different human-written captions.
- **Preprocessing:** Applied text tokenization, vocabulary building, and image resizing/normalization.
- **Evaluation:** Performance is measured using the **BLEU Score** metric to quantify the similarity between the machine-generated captions and human reference captions.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/jatin-42/image-captioning-cnn-rnn.git](https://github.com/jatin-42/image-captioning-cnn-rnn.git)

2. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas matplotlib

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook image_captioning.ipynb

4. Run the notebook cells to train the model or load saved weights for inference on new images.
