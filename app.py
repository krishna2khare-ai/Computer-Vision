import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import os
from skimage import measure, morphology
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from collections import defaultdict
import warnings
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           average_precision_score, roc_curve, auc, 
                           classification_report, precision_recall_curve)
from scipy.spatial.distance import dice
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
import pickle
import hashlib

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Constants
EVALUATION_CACHE_FILE = "evaluation_cache.pkl"

# Title and description
st.set_page_config(layout="wide")
st.title("Retinal CVD Analysis System")
st.write("""
A comprehensive tool for retinal image analysis and cardiovascular disease risk prediction.
Upload a retinal fundus image to get detailed metrics and visualizations.
""")

class RetinalAnalyzer:
    """Core analysis engine with metrics calculation"""
    
    def __init__(self):
        self.img_size = (256, 256)
        self.feature_names = [
            'VesselDensity', 'Tortuosity', 'CupDiscRatio',
            'LesionCount', 'EdgeComplexity', 'FractalDim',
            'AVR', 'CRAE', 'CRVE'
        ]
        
    def load_image(self, file_path):
        """Load and preprocess image"""
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Failed to load image")
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        img = cv2.resize(img, self.img_size)
        return img

    def process_image(self, img):
        """Complete image processing pipeline"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Vessel extraction
        kernel = np.ones((3,3), np.uint8)
        opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
        vessels = cv2.absdiff(enhanced, opened)
        
        # Optic disc detection
        disc_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT, dp=1.2,
                                 minDist=50, param1=80, param2=35,
                                 minRadius=30, maxRadius=80)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0,:]:
                cv2.circle(disc_img, (x,y), r, (0,255,0), 2)
        
        # Lesion detection
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        _, a_channel, b_channel = cv2.split(lab)
        _, bright = cv2.threshold(b_channel, 160, 255, cv2.THRESH_BINARY)
        dark = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)
        lesions = cv2.bitwise_or(bright, dark)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        return {
            'original': img,
            'enhanced': enhanced,
            'vessels': vessels,
            'optic_disc': disc_img,
            'lesions': lesions,
            'edges': edges
        }

    def calculate_metrics(self, processed, ground_truth=None):
        """Calculate all image quality and feature metrics"""
        metrics = defaultdict(dict)
        
        # Image quality metrics
        for name in ['vessels', 'optic_disc', 'lesions', 'edges']:
            metrics['quality'][f'{name}_ssim'] = ssim(
                cv2.cvtColor(processed['original'], cv2.COLOR_BGR2GRAY),
                processed[name] if name != 'optic_disc' else cv2.cvtColor(processed[name], cv2.COLOR_BGR2GRAY),
                data_range=255
            )
            metrics['quality'][f'{name}_psnr'] = psnr(
                cv2.cvtColor(processed['original'], cv2.COLOR_BGR2GRAY),
                processed[name] if name != 'optic_disc' else cv2.cvtColor(processed[name], cv2.COLOR_BGR2GRAY),
                data_range=255
            )
        
        # Vessel metrics
        skeleton = morphology.skeletonize(processed['vessels'] > 0)
        labeled = measure.label(skeleton)
        props = measure.regionprops(labeled)
        
        metrics['features']['vessel_density'] = np.sum(processed['vessels'] > 0) / processed['vessels'].size
        metrics['features']['vessel_tortuosity'] = np.mean([
            (p.perimeter/(2*np.sqrt(np.pi*p.area)))-1 for p in props if p.perimeter > 0
        ]) if props else 0
        
        # Lesion metrics
        metrics['features']['lesion_count'] = np.sum(processed['lesions'] > 0)
        
        # Edge metrics
        metrics['features']['edge_complexity'] = np.sum(processed['edges'] > 0) / processed['edges'].size
        
        # # Computer vision metrics if ground truth available
        # if ground_truth:
        #     cv_metrics = self.calculate_cv_metrics(processed, ground_truth)
        #     metrics.update(cv_metrics)
        
        return metrics

    # def calculate_cv_metrics(self, processed, ground_truth):
    #     """Calculate computer vision specific metrics"""
    #     cv_metrics = {}
        
    #     # Dice coefficient for each component
    #     for name in ['vessels', 'lesions', 'edges']:
    #         if name in processed and name in ground_truth:
    #             intersection = np.logical_and(processed[name] > 0, ground_truth[name] > 0).sum()
    #             union = (processed[name] > 0).sum() + (ground_truth[name] > 0).sum()
    #             cv_metrics[f'{name}_dice'] = 2 * intersection / union if union > 0 else 0.0
        
    #     return {'cv_metrics': cv_metrics}

    def generate_features(self, processed, metrics):
        """Generate feature vector for prediction"""
        return {
            'VesselDensity': metrics['features']['vessel_density'],
            'Tortuosity': metrics['features']['vessel_tortuosity'],
            'CupDiscRatio': self._calculate_cup_disc_ratio(processed['optic_disc']),
            'LesionCount': metrics['features']['lesion_count'],
            'EdgeComplexity': metrics['features']['edge_complexity'],
            'FractalDim': self._calculate_fractal_dim(processed['vessels']),
            'AVR': self._calculate_avr(processed['vessels']),
            'CRAE': self._calculate_vessel_metrics(processed['vessels'])[0],
            'CRVE': self._calculate_vessel_metrics(processed['vessels'])[1]
        }

    def _calculate_cup_disc_ratio(self, disc_img):
        """Calculate cup-to-disc ratio"""
        gray = cv2.cvtColor(disc_img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                 minDist=50, param1=80, param2=35,
                                 minRadius=30, maxRadius=80)
        if circles is None:
            return 0
        return np.pi * (circles[0,0,2]**2) / gray.size

    def _calculate_fractal_dim(self, image):
        """Calculate fractal dimension with comprehensive error handling"""
        try:
            # Create binary image
            binary = (image > 0.5*np.max(image)).astype(np.uint8)
            
            # Validate we have enough structure to analyze
            if np.sum(binary) < 100:  # Minimum number of pixels
                return 1.0  # Default value for simple structures
            
            # Generate box sizes (powers of 2)
            sizes = 2**np.arange(1, int(np.log2(min(binary.shape))))
            
            # Box counting
            counts = []
            for size in sizes:
                box = binary[::size, ::size]
                count = np.count_nonzero(box)
                if count > 0:  # Only use non-empty boxes
                    counts.append(count)
            
            # Need at least 3 points for regression
            if len(counts) < 3:
                return 1.0
            
            # Linear regression on log-log plot
            coeffs = np.polyfit(np.log(sizes[:len(counts)]), np.log(counts), 1)
            return float(-coeffs[0])  # Ensure we return a float
            
        except Exception as e:
            print(f"Fractal dimension calculation error: {str(e)}")
            return 1.0  # Fallback value

    def _calculate_avr(self, vessel_image):
        """Calculate arteriole-to-venule ratio"""
        contours, _ = cv2.findContours(vessel_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        diameters = [cv2.minEnclosingCircle(cnt)[1]*2 for cnt in contours]
        if len(diameters) < 2:
            return 0.5
        half = len(diameters)//2
        return np.mean(sorted(diameters)[:half])/np.mean(sorted(diameters)[half:])

    def _calculate_vessel_metrics(self, vessel_image):
        """Calculate CRAE and CRVE"""
        contours, _ = cv2.findContours(vessel_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        diameters = [cv2.minEnclosingCircle(cnt)[1]*2 for cnt in contours]
        if len(diameters) < 2:
            return (0.0, 0.0)
        half = len(diameters)//2
        return (
            np.mean(sorted(diameters)[:half]),
            np.mean(sorted(diameters)[half:])
        )

class CVDClassifier:
    """Prediction subsystem"""
    
    def __init__(self, model_path="best_model.keras", scaler_path="feature_scaler.save"):
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            scaler_data = joblib.load(scaler_path)
            self.scaler = scaler_data['scaler'] if isinstance(scaler_data, dict) else scaler_data
            self.analyzer = RetinalAnalyzer()
        except Exception as e:
            st.error(f"Failed to load model or scaler: {str(e)}")
            self.model = None
            self.scaler = None

    def predict(self, image_path, ground_truth=None):
        """Complete prediction pipeline"""
        try:
            if self.model is None or self.scaler is None:
                return {'status': 'error', 'message': 'Model not loaded'}
                
            # Process image
            img = self.analyzer.load_image(image_path)
            processed = self.analyzer.process_image(img)
            metrics = self.analyzer.calculate_metrics(processed, ground_truth)
            features = self.analyzer.generate_features(processed, metrics)
            
            # Prepare input
            feature_vector = np.array([features[col] for col in self.analyzer.feature_names]).reshape(1,-1)
            scaled_features = self.scaler.transform(feature_vector)
            img_array = np.expand_dims(img.astype(np.float32)/255.0, axis=0)
            
            # Predict
            proba = self.model.predict([img_array, scaled_features], verbose=0)[0][0]
            
            return {
                'status': 'success',
                'prediction': 'CVD Positive' if proba > 0.5 else 'CVD Negative',
                'probability': float(proba),
                'processed': processed,
                'metrics': metrics,
                'features': features,
                'original_img': cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

def display_metrics(result, ground_truth=None):
    """Display all metrics and visualizations"""
    
    # Prediction result
    st.subheader("Prediction Result")
    cols = st.columns(3)
    cols[0].metric("Prediction", result['prediction'])
    cols[1].metric("Probability", f"{result['probability']:.1%}")
    
    # if ground_truth and 'cv_metrics' in result['metrics']:
    #     cols[2].metric("Vessel Dice", f"{result['metrics']['cv_metrics'].get('vessels_dice', 0):.3f}")
    
    # Processing pipeline visualization
    st.subheader("Image Processing Pipeline")
    cols = st.columns(3)
    components = {
        'Original': result['original_img'],
        'Enhanced': result['processed']['enhanced'],
        'Vessels': result['processed']['vessels'],
        'Optic Disc': result['processed']['optic_disc'],
        'Lesions': result['processed']['lesions'],
        'Edges': result['processed']['edges']
    }
    
    for i, (name, img) in enumerate(components.items()):
        cols[i%3].image(img, caption=name, use_container_width=True)
    
    # Metrics tabs
    tab_names = ["Quality Metrics", "Feature Analysis", "Clinical Insights"]
    if ground_truth:
        tab_names.append("Performance Metrics")
    
    tabs = st.tabs(tab_names)
    
    with tabs[0]:  # Quality Metrics
        st.subheader("Image Quality Metrics")
        quality_df = pd.DataFrame.from_dict(result['metrics']['quality'], orient='index', columns=['Value'])
        st.dataframe(quality_df.style.format("{:.3f}"))
        
        fig, ax = plt.subplots(figsize=(10,4))
        quality_df.plot(kind='bar', ax=ax)
        ax.set_title("Image Quality Metrics Comparison")
        ax.set_ylabel("Score")
        st.pyplot(fig)
    
    with tabs[1]:  # Feature Analysis
        st.subheader("Feature Analysis")
        
        # Feature values
        feature_df = pd.DataFrame.from_dict(result['features'], orient='index', columns=['Value'])
        feature_df.fillna(0.0, inplace=True)  # Handle None values
        st.dataframe(feature_df.style.format("{:.3f}"))
        
        # Feature distributions
        fig, axes = plt.subplots(3, 3, figsize=(12,10))
        for i, (feat, val) in enumerate(result['features'].items()):
            ax = axes[i//3, i%3]
            ax.barh([0], [val], color='skyblue')
            ax.set_title(feat)
            ax.set_xlim(0, max(1, val*1.5))
        plt.tight_layout()
        st.pyplot(fig)
        
        # Vessel characteristics
        st.write("**Vessel Characteristics**")
        fig, ax = plt.subplots()
        ax.scatter(
            result['features']['VesselDensity'],
            result['features']['Tortuosity'],
            s=100, color='red'
        )
        ax.set_xlabel("Vessel Density")
        ax.set_ylabel("Vessel Tortuosity")
        ax.set_title("Vessel Health Indicators")
        st.pyplot(fig)
    
    with tabs[2]:  # Clinical Insights
        st.subheader("Clinical Interpretation")
        
        # Risk factors
        risk_factors = []
        if result['features']['VesselDensity'] < 1:
            risk_factors.append("Low vessel density (possible ischemia)")
        if result['features']['Tortuosity'] < 0.1:
            risk_factors.append("High vessel tortuosity (possible hypertension)")
        if result['features']['LesionCount'] > 500:
            risk_factors.append("Significant lesions detected")
        
        if risk_factors:
            st.warning("**Potential Risk Factors Detected:**")
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.success("No significant risk factors detected")
        
        # Feature explanations
        with st.expander("Feature Descriptions"):
            st.write("""
            - **Vessel Density**: Proportion of image area containing blood vessels
            - **Tortuosity**: Measure of blood vessel twisting/curvature
            - **Cup-Disc Ratio**: Optic nerve head structural measurement
            - **Lesion Count**: Number of detected hemorrhages/exudates
            - **Edge Complexity**: Retinal layer boundary irregularity
            - **Fractal Dimension**: Vascular branching complexity
            - **AVR**: Arteriole-to-Venule diameter ratio
            - **CRAE/CRVE**: Central retinal artery/vein equivalent diameters
            """)
    
    if ground_truth and len(tabs) > 3:  # Performance Metrics
        with tabs[3]:
            st.subheader("Performance Metrics")
            
            # Calculate classification metrics
            y_true = ground_truth['label']
            y_pred = result['prediction'] == 'CVD Positive'
            y_score = result['probability']
            
            # Classification report
            st.write("### Classification Report")
            report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"))
            
            # Confusion matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted Negative', 'Predicted Positive'],
                       yticklabels=['Actual Negative', 'Actual Positive'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # ROC Curve
            st.write("### ROC Curve")
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            st.pyplot(fig)

def load_test_dataset(dataset_path):
    """Load test dataset with labels from directory structure"""
    test_data = {}
    dataset_path = Path(dataset_path)

    # Check if directory exists
    if not dataset_path.exists():
        st.error(f"Directory not found: {dataset_path}")
        return test_data

    # Define positive and negative subdirectories using Path
    positive_dir = dataset_path / "CVD_Positive"
    negative_dir = dataset_path / "CVD_Negative"

    # Check if subdirectories exist
    if not positive_dir.exists() or not negative_dir.exists():
        st.error(f"Test directory must contain 'CVD_Positive' and 'CVD_Negative' subfolders")
        return test_data

    # Load positive cases
    for img_path in positive_dir.glob('*.*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            test_data[str(img_path)] = {'label': True}

    # Load negative cases
    for img_path in negative_dir.glob('*.*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            test_data[str(img_path)] = {'label': False}

    if not test_data:
        st.warning("No images found in the test directory")
    
    return test_data

def get_dataset_hash(dataset_path):
    """Generate a hash of the dataset directory to detect changes"""
    hash_obj = hashlib.md5()
    dataset_path = Path(dataset_path)
    
    # Hash the directory structure and file names (not contents for speed)
    for root, dirs, files in os.walk(dataset_path):
        for file in sorted(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                hash_obj.update(file.encode())
    
    return hash_obj.hexdigest()

@st.cache_resource(show_spinner=False)
def load_or_run_evaluation(dataset_path):
    """Load cached evaluation or run fresh evaluation"""
    dataset_hash = get_dataset_hash(dataset_path)
    cache_data = {}
    
    # Try to load existing cache
    if os.path.exists(EVALUATION_CACHE_FILE):
        try:
            with open(EVALUATION_CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                if cache_data.get('dataset_hash') == dataset_hash:
                    return cache_data['results']
        except:
            pass
    
    # Run fresh evaluation if no cache exists
    test_data = load_test_dataset(dataset_path)
    if not test_data:
        return None
    
    results = {
        'y_true': [],
        'y_pred': [],
        'y_scores': [],
        # 'dice_scores': [],
        'image_paths': []
    }
    
    classifier = CVDClassifier()
    if classifier.model is None:
        return None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_images = len(test_data)
    
    for i, (img_path, gt) in enumerate(test_data.items()):
        status_text.text(f"Processing image {i+1}/{total_images}: {Path(img_path).name}")
        progress_bar.progress((i+1)/total_images)
        
        try:
            result = classifier.predict(img_path, gt)
            if result['status'] == 'success':
                results['y_true'].append(gt['label'])
                results['y_pred'].append(result['prediction'] == 'CVD Positive')
                results['y_scores'].append(result['probability'])
                results['image_paths'].append(img_path)
                # if 'cv_metrics' in result['metrics']:
                #     results['dice_scores'].append(result['metrics']['cv_metrics'].get('vessels_dice', 0))
        except Exception as e:
            st.warning(f"Error processing {img_path}: {str(e)}")
            continue
    
    # Save to cache
    cache_data = {
        'dataset_hash': dataset_hash,
        'results': results
    }
    with open(EVALUATION_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return results

def display_evaluation_results(results):
    """Display cached evaluation results"""
    if not results or not results['y_true']:
        st.error("No evaluation results available")
        return
    
    # Generate performance report
    st.subheader("Model Evaluation Results (Cached)")
    
    # Classification report
    st.write("### Classification Report")
    report = classification_report(results['y_true'], results['y_pred'], 
                                 target_names=['Negative', 'Positive'], 
                                 output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))
    
    # Confusion matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Predicted Negative', 'Predicted Positive'],
               yticklabels=['Actual Negative', 'Actual Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # ROC Curve
    st.write("### ROC Curve")
    fpr, tpr, _ = roc_curve(results['y_true'], results['y_scores'])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    st.pyplot(fig)
    
    # Precision-Recall Curve
    st.write("### Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(results['y_true'], results['y_scores'])
    avg_precision = average_precision_score(results['y_true'], results['y_scores'])
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    st.pyplot(fig)
    
    # Metrics summary
    st.write("### Metrics Summary")
    metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 
              'ROC AUC', 'Average Precision'],
    'Value': [
        accuracy_score(results['y_true'], results['y_pred']),
        precision_score(results['y_true'], results['y_pred']),
        recall_score(results['y_true'], results['y_pred']),
        f1_score(results['y_true'], results['y_pred']),
        roc_auc,
        avg_precision
    ]
})
    st.dataframe(metrics_df.style.format({'Value': '{:.3f}'}))

# Main application
def main():
    mode = st.sidebar.radio("Select Mode", ["Single Image Analysis", "Model Evaluation"])
    
    if mode == "Single Image Analysis":
        uploaded_file = st.file_uploader("Upload retinal image", type=["jpg","jpeg","png"])
        
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                # Save temp file
                temp_path = "temp_retinal.jpg"
                Image.open(uploaded_file).save(temp_path)
                
                # Load classifier and predict
                classifier = CVDClassifier()
                result = classifier.predict(temp_path)
                
                # Clean up
                os.remove(temp_path)
                
                if result['status'] == 'success':
                    display_metrics(result)
                else:
                    st.error(f"Analysis failed: {result['message']}")
    else:
        st.info("Model evaluation requires ground truth data. Results are cached for faster loading.")
        
        default_path = r"C:\Users\mgree\Downloads\CVD_SPLIT_DATASET\test"
        dataset_path = st.text_input("Enter path to test dataset", default_path)
        
        if st.button("Run Evaluation"):
            if not dataset_path:
                st.warning("Please enter a valid dataset path")
                return
                
            results = load_or_run_evaluation(dataset_path)
            if results:
                display_evaluation_results(results)
            else:
                st.error("Evaluation failed or no test data found")
                
        # Add button to clear cache if needed
        if st.button("Clear Cache and Re-run Evaluation"):
            if os.path.exists(EVALUATION_CACHE_FILE):
                os.remove(EVALUATION_CACHE_FILE)
            st.experimental_rerun()

if __name__ == "__main__":
    main()