import argparse
from PIL import Image
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from joblib import load

# Load trained classifier
clf = load('model/human_animal_classifier.joblib')

# Load CLIP model for feature extraction
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", use_fast=False
)

# Define your class labels based on your training folder order
label_classes = ['Cat', 'Cow', 'Deer', 'Dog', 'Goat', 'Human', 'Sheep']

def extract_features_pil(image):
    """Extract CLIP features from a PIL image."""
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()

def predict_class(features):
    """Predict class label and confidence using trained classifier."""
    prob = clf.predict_proba([features])[0]
    predicted_idx = prob.argmax()
    confidence = prob[predicted_idx]
    label = label_classes[predicted_idx]
    return label, confidence

def detect(path, is_image):
    if is_image:
        
        image = Image.open(path).convert("RGB")
        features = extract_features_pil(image)
        label, confidence = predict_class(features)
      
        print(f"[IMAGE] Detected: {label} (Confidence: {confidence * 100:.2f}%)")
    else:
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

           
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            features = extract_features_pil(image)
            label, confidence = predict_class(features)

            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Detection', frame)

    
            if confidence > 0.80:
                print(f"[VIDEO] ALERT: Detected {label} (Confidence: {confidence * 100:.2f}%)")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human & Animal Classifier")
    parser.add_argument("--path", type=str, required=True, help="Path to image or video file")
    parser.add_argument('--is_image', type=str, default="True", help="True for image, False for video.")
    # parser.add_argument("--is_image", type=bool, required=True, help="True for image, False for video")

    args = parser.parse_args()

    # Convert string to actual boolean
    is_image = args.is_image.lower() == "true"

    detect(args.path, is_image=is_image)