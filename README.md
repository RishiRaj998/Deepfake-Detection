# Deepfake-Detection
# Detecting Deepfakes Across Modalities: A Multi-Modal Approach to Video, Audio, Image, and Text Analysis

## Project Overview

This dissertation project investigates and implements a robust defense mechanism against the rapidly expanding threat of synthetic media across four critical domains: **Text, Image, Video, and Audio**. The core deliverable is a **Unified Multi-Modal Hybrid Architecture** designed to proactively counter the increasing scale and sophistication of deepfakes.

### 👤 Author and Supervision
*   **Author:** Rishi Raj Sharma (Roll No: 2402519)
*   **Degree:** Master of Science in Computer Science
*   **Supervisor:** Dr. Prem Sewak Sudhish
*   **Institution:** Dayalbagh Educational Institute, AGRA (UP)

## 🎯 Motivation and Critical Problem

The fundamental motivation stems from the failure of existing deepfake detection models (across all modalities) to **generalize robustly against rapidly evolving, multi-modal "in-the-wild" attacks**.

A novel and overlooked risk addressed by this work is the **efficient personalized text generation** capability of modern Large Language Models (LLMs). Using techniques like **Low-Rank Adaptation (LoRA)**, malicious actors can **credibly imitate an individual’s unique writing style, tone, and language** using surprisingly small amounts of data. This enables highly persuasive attacks, such as **personalized phishing emails** or fraudulent social media accounts, a threat that is distinct from, and often neglected by, current legal frameworks focusing primarily on image/video deepfakes.

## ⚙️ Hybrid System Architecture

The project utilizes a **Hybrid Multi-Modal Architecture** (Figure 1 in Dissertation). This system operates as a specialized **triage** mechanism, routing input media to the appropriate pre-trained model pipeline. It strategically combines **Classical Machine Learning (ML)** models with **Advanced Deep Learning (DL)** models to maximize detection performance in each modality.

The entire framework is integrated into a user-friendly web interface built with **Streamlit**.

## 📈 Modality Performance Highlights

| Modality | Core Architecture / Methodology | Key Dataset(s) | Peak Accuracy / Metric | Key Features Detected |
| :--- | :--- | :--- | :--- | :--- |
| **Audio** | **Hybrid LSTM-KNN System** (LSTM for feature learning, KNN for classification) | `DEEP-VOICE` | **96% Accuracy** (surpassing MLP baseline of 90%) | Temporal voice patterns |
| **Video** | **Vision Transformer (ViT)** adapted to sequences of extracted frames | `Model_One_Training` | **93% Accuracy** | Spatial relationships, Temporal Forgery Artifacts (e.g., Facial Feature Drift-FFD) |
| **Image** | **Vision Transformer (ViT)** | Deepfake Image Detection Dataset | **82% Accuracy** (improved over ResNet-50 baseline of 75%) | Long-range spatial relationships, fine-grained deepfake artifacts |
| **Text** | **Classical ML** (Random Forest, Gradient-Boost) on **TF-IDF Vectorization** | Deepfake News Dataset | **90.8% Accuracy** | Semantic subtleties and stylistic patterns (using **FastText embeddings**) |

## 🚧 Critical Challenges and Limitations

The most significant constraint encountered was the instability within the Text Detection pipeline:

*   **Serialization Mismatch:** The text detection model exhibited persistent **unstable and inconsistent predictions** during deployment. This structural error was successfully isolated to a **serialization mismatch** in the vectorization inference phase, indicating highly complex dependency management issues when integrating disparate models.
*   **Generalization Gap:** The performance of models, even with high laboratory accuracy, struggled to generalize robustly against contemporary **"in-the-wild" deepfakes**. To address this, testing utilized benchmarks inspired by the **Deepfake-Eval-2024** dataset.
*   **Resource Constraints:** Further exploration of advanced video architectures, such as the **R(2+1)D model**, was limited due to high computational demands and the exhaustion of the available **T4 GPU quota**.

## 🗓️ Future Work and Recommendations

1.  **Text Pipeline Re-engineering:** Immediate re-engineering and re-training of the text detection pipeline are required to resolve the serialization mismatch and ensure stable predictions.
2.  **Generalization Improvement:** Expand dataset diversity and model training (e.g., training models on **new datasets**) to develop more **content-agnostic** defenses capable of handling emerging deepfake techniques.
3.  **Policy Advocacy:** Given the proven viability of text imitation attacks, the research community must urge **systematic and legal protections** for victims of text deepfakes, as current legislation often excludes this modality.

## 🛠️ Installation and Setup (Conceptual)

To run this project locally, a compatible environment with Python 3.x is required.

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```

2.  **Install Dependencies:**
    The project relies on PyTorch (for implementation and training), NumPy, Pandas, Scikit-learn, OpenCV, TQDM, TorchSummary, and the **Streamlit** framework for the interface.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Large model weights and datasets like `Model_One_Training` and `DEEP-VOICE` may require manual download or Git LFS setup.)*

3.  **Run the Multi-Modal Interface:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your browser, enabling access to Text, Audio, Image, and Video deepfake detection modules.
```
