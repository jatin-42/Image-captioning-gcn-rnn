# 🕸️ Image Captioning with Graph Convolutional Networks (GCN)

An advanced Image Captioning system that goes beyond pixel analysis. Unlike traditional CNN-only models, this project utilizes **Graph Convolutional Networks (GCN)** to model the semantic relationships between objects in an image (Scene Graphs) for more accurate caption generation.

## 🧠 The Architecture (GCN + LSTM)
This model understands the "context" of an image by constructing a graph structure:
1.  **Visual Graph Construction:** The image is processed to identify objects (Nodes) and their spatial/semantic relationships (Edges).
2.  **Graph Encoder (GCN):** A Graph Convolutional Network processes this structure to create "semantic embeddings" that capture how objects interact (e.g., "Man" --*holding*--> "Bat").
3.  **Decoder (LSTM/RNN):** These relation-aware embeddings are fed into an LSTM to generate natural language descriptions.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow
- **Graph Neural Networks:** GCN (Graph Convolutional Network)
- **Computer Vision:** CNN (for initial object detection)
- **Language:** Python
- **Data:** Flickr8k 

## 🚀 Why GCN?
Traditional CNNs often list objects ("dog", "frisbee", "park") but miss the interaction. By using a GCN, this model explicitly learns the relationship ("dog *catching* frisbee"), resulting in more descriptive and grammatically complex captions.

## 📂 Project Structure
- `image_captioning.ipynb`: The complete pipeline, including data preprocessing, GCN model definition, training loop, and inference.
- `notebooks/`: (Optional) Additional experiments and visualization of generated scene graphs.

## 📊 Dataset & Performance
- **Dataset:** Trained on the **Flickr8k dataset**, consisting of 8,000 images, each paired with 5 different human-written captions.
- **Preprocessing:** Applied text tokenization, vocabulary building, and image resizing/normalization.
- **Evaluation:** Performance is measured using the **BLEU Score** metric to quantify the similarity between the machine-generated captions and human reference captions.

## 📂 Dataset Instructions
This model works with the **Flickr8k Dataset**.
Due to file size limits, the image data is not included in this repository.

**Download Instructions:**
1. Download the dataset from [Kaggle - Flickr8k Dataset](https://www.kaggle.com/adityajn105/flickr8k).
2. Unzip the folder.
3. Place the `Images` folder inside the root directory of this project.

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
