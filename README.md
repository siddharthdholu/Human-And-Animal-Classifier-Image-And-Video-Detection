
# 🧠 Human & Animal Classifier — Image & Video Detection
This project demonstrates a simple AI-based detection system that identifies and classifies **humans** and **animals** in images or videos using OpenAI’s `CLIP` model for feature extraction and a custom-trained `Logistic Regression` classifier.

## 📁 Project Structure

    ASSIGNMENT/
    ├── data/
    │   ├── Cat/
    │   ├── Cow/
    │   ├── Deer/
    │   ├── Dog/
    │   ├── Goat/
    │   ├── Human/
    │   └── Sheep/
    ├── model/
    │   └── human_animal_classifier.joblib
    ├── main.ipynb
    ├── detect.py
    └── requirements.txt

## 💡 Project Overview
This solution is designed to:

1.  **Detect and differentiate** between humans and animals in an image or video.
    
2.  **Classify** animals into specific categories:  
    `Cat`, `Cow`, `Deer`, `Dog`, `Goat`, `Human`, `Sheep`.
    
3.  **Trigger alerts** via console or video overlay whenever an object is detected with high confidence.

## ⚙️ Technical Details
| Feature | Description |
|--|--|
|**Language**|Python 3.10+  |
|**Feature Extraction**|OpenAI `CLIP` model (`clip-vit-base-patch32`) via `transformers`
|**Classifier Model**|`LogisticRegression` (trained using `scikit-learn`)
|**Model Saving Format**|`.joblib`
|**Detection Modes**|Image & Video compatible
|**Threshold for Alerts**|80% Confidence
**Input Types**|Images & Videos

## 🐍 Installation

 1. Clone or download this repository.
 2. Install dependencies:
 ```bash
 pip install -r requirements.txt
 ```

## 🖼️ How to Use — Image or Video Detection
You can pass either an image or video path to the `detect_from_path()` function:
```python
# For single image:
detect_from_path("path/to/image.jpg", is_image=True)

# For video:
detect_from_path("path/to/video.mp4", is_image=False)
```
## 🔍 How to Use `detect.py` for Detection (Via Terminal)
This script allows you to test the trained model on **images or videos**.
### 💡 Usage:
```bash
python detect.py --path <file_path> --is_image <True/False>
```
| Argument | Description | Example |
|---|---|---|
|`--path`|Path to your image or video file.|`data_samples/sample.jpg`
|`--is_image`|Set `True` if the file is an image, `False` if a video.|`True`

### 💡 Example Commands:
```bash
# For image
python detect.py --path data_samples/sample.jpg --is_image True
# For video
python detect.py --path data_samples/sample_video.mp4 --is_image False
```
### ✅ When the script runs:
-   It will print the detected class label and confidence.
-   If it’s a video, the detection will be displayed frame-by-frame with real-time predictions.
-   To exit video detection, press **`Q`** on your keyboard.

## 🔔 Alert Mechanism
-   **Images:** Prints detection result to console.
-   **Videos:** Displays detected class and confidence on video frames and triggers console alerts when confidence ≥ 80%.

## 🏆 Sample Output
-   `[IMAGE] Detected: Human (Confidence: 95.63%)`
-   `[VIDEO] ALERT: Detected Dog (Confidence: 91.47%)`

## 📌 Notes
-   The model is trained on static images only, but the same approach is used to predict each frame in a video.
-   You can extend the class list by adding more folders under `/data` and retraining.

## 📂 Dependencies
```bash
torch>=2.0  
transformers>=4.40  
scikit-learn>=1.4  
opencv-python>=4.8  
Pillow>=10.0  
joblib>=1.3  
```

## ✅ Conclusion
This project meets the following criteria:

-   🔍 **Human/Animal Detection**
-   🐾 **Animal Classification**
-   ⚠️ **Alert Triggering System**
-   📂 **Tested on Image & Video Input**