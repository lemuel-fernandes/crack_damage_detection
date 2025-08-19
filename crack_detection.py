#!/usr/bin/env python3
"""
Run Crack Detection on building.jpg
This script will analyze your building image for cracks and damage.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class CrackDetectionSystem:
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.results = []
        
    def get_default_config(self):
        return {
            'gaussian_kernel_size': (5, 5),
            'gaussian_sigma': 1.0,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            'canny_low_threshold': 50,
            'canny_high_threshold': 150,
            'canny_aperture': 3,
            'morph_kernel_size': (3, 3),
            'closing_iterations': 2,
            'opening_iterations': 1,
            'min_contour_area': 100,
            'max_contour_area': 50000,
            'min_aspect_ratio': 2.0,
            'max_width_height_ratio': 0.3,
            'crack_length_threshold': 100,
            'crack_width_threshold': 10,
            'pothole_circularity_threshold': 0.3,
            'output_dir': 'building_analysis_results',
            'visualization': True
        }
    
    def preprocess_image(self, image):
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, self.config['gaussian_kernel_size'], self.config['gaussian_sigma'])
        
        # Bilateral filter to preserve edges
        filtered = cv2.bilateralFilter(blurred, self.config['bilateral_d'], 
                                     self.config['bilateral_sigma_color'], 
                                     self.config['bilateral_sigma_space'])
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        return enhanced
    
    def detect_edges(self, image):
        # Canny edge detection
        edges_canny = cv2.Canny(image, self.config['canny_low_threshold'], 
                               self.config['canny_high_threshold'], 
                               apertureSize=self.config['canny_aperture'])
        
        # Sobel edge detection
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(grad_x**2 + grad_y**2)
        edges_sobel = np.uint8(edges_sobel / edges_sobel.max() * 255)
        _, edges_sobel = cv2.threshold(edges_sobel, 50, 255, cv2.THRESH_BINARY)
        
        # Combine edge maps
        combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
        return combined_edges
    
    def apply_morphological_operations(self, edges):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config['morph_kernel_size'])
        
        # Closing to connect nearby edges
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, 
                                 iterations=self.config['closing_iterations'])
        
        # Opening to remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, 
                                 iterations=self.config['opening_iterations'])
        return opened
    
    def extract_contours(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config['min_contour_area'] or area > self.config['max_contour_area']:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            width_height_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            
            if (aspect_ratio >= self.config['min_aspect_ratio'] and 
                width_height_ratio <= self.config['max_width_height_ratio']):
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def classify_damage(self, contour, image_shape):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        length = max(w, h)
        width = min(w, h)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Classification logic
        if aspect_ratio > self.config['min_aspect_ratio']:
            if length > self.config['crack_length_threshold']:
                damage_type = "Long Crack"
                severity = "High" if width > self.config['crack_width_threshold'] else "Medium"
            else:
                damage_type = "Short Crack"
                severity = "Medium" if width > 5 else "Low"
        elif circularity > self.config['pothole_circularity_threshold']:
            damage_type = "Pothole"
            severity = "High" if area > 1000 else "Medium"
        else:
            damage_type = "Surface Damage"
            severity = "Medium"
        
        return {
            'type': damage_type,
            'severity': severity,
            'area': area,
            'perimeter': perimeter,
            'length': length,
            'width': width,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'bounding_box': (x, y, w, h),
            'center': (x + w//2, y + h//2)
        }
    
    def process_image(self, image_path):
        print(f"Loading image: {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"ERROR: Image file '{image_path}' not found!")
            print("Make sure the file is in the same directory as this script.")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: Could not load image '{image_path}'")
            print("Make sure the file is a valid image format (jpg, png, etc.)")
            return None
        
        print(f"Image loaded successfully! Dimensions: {image.shape}")
        
        original_image = image.copy()
        image_name = os.path.basename(image_path)
        
        # Processing steps
        print("Step 1: Preprocessing image...")
        preprocessed = self.preprocess_image(image)
        
        print("Step 2: Detecting edges...")
        edges = self.detect_edges(preprocessed)
        
        print("Step 3: Applying morphological operations...")
        cleaned_edges = self.apply_morphological_operations(edges)
        
        print("Step 4: Extracting contours...")
        contours = self.extract_contours(cleaned_edges)
        print(f"Found {len(contours)} potential damage areas")
        
        print("Step 5: Classifying damage...")
        damages = []
        for contour in contours:
            damage_info = self.classify_damage(contour, image.shape[:2])
            damages.append(damage_info)
        
        # Results
        result = {
            'image_name': image_name,
            'image_path': image_path,
            'image_shape': image.shape,
            'total_damages': len(damages),
            'damages': damages,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        print(f"Analysis complete! Found {len(damages)} damage areas")
        
        # Create visualizations
        self.create_visualizations(original_image, preprocessed, edges, cleaned_edges, 
                                 contours, damages, image_name)
        
        # Print summary
        self.print_summary(result)
        
        return result
    
    def create_visualizations(self, original, preprocessed, edges, cleaned_edges, 
                            contours, damages, image_name):
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Create analysis figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Building Crack Detection Analysis: {image_name}', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Building Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Preprocessed image
        axes[0, 1].imshow(preprocessed, cmap='gray')
        axes[0, 1].set_title('Preprocessed (Enhanced)', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Edge detection
        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title('Edge Detection (Canny + Sobel)', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Cleaned edges
        axes[1, 0].imshow(cleaned_edges, cmap='gray')
        axes[1, 0].set_title('Morphological Cleaning', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Detected contours
        contour_image = original.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        axes[1, 1].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Detected Features ({len(contours)})', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Final annotated results
        result_image = original.copy()
        self.draw_annotations(result_image, damages)
        axes[1, 2].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'Final Analysis ({len(damages)} damages)', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save the analysis plot
        output_path = os.path.join(self.config['output_dir'], f'{os.path.splitext(image_name)[0]}_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Analysis saved to: {output_path}")
        
        # Also save the annotated result separately
        result_path = os.path.join(self.config['output_dir'], f'{os.path.splitext(image_name)[0]}_annotated.jpg')
        cv2.imwrite(result_path, result_image)
        print(f"Annotated image saved to: {result_path}")
        
        plt.show()  # Display the plot
    
    def draw_annotations(self, image, damages):
        # Color mapping for different damage types
        color_map = {
            'Long Crack': (0, 0, 255),      # Red
            'Short Crack': (0, 165, 255),   # Orange
            'Pothole': (255, 0, 0),         # Blue
            'Surface Damage': (0, 255, 255) # Yellow
        }
        
        for i, damage in enumerate(damages):
            x, y, w, h = damage['bounding_box']
            damage_type = damage['type']
            severity = damage['severity']
            
            color = color_map.get(damage_type, (128, 128, 128))
            thickness = 3 if severity == 'High' else 2
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            label = f"{damage_type} ({severity})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Label text
            cv2.putText(image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Damage number
            center_x, center_y = damage['center']
            cv2.putText(image, str(i + 1), (center_x - 10, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def print_summary(self, result):
        print("\n" + "="*60)
        print("BUILDING DAMAGE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Image: {result['image_name']}")
        print(f"Dimensions: {result['image_shape'][1]} x {result['image_shape'][0]} pixels")
        print(f"Total Damage Areas Found: {result['total_damages']}")
        
        if result['damages']:
            print("\nDetailed Findings:")
            print("-" * 40)
            
            # Count by type and severity
            type_counts = {}
            severity_counts = {'Low': 0, 'Medium': 0, 'High': 0}
            
            for i, damage in enumerate(result['damages'], 1):
                dtype = damage['type']
                severity = damage['severity']
                
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
                severity_counts[severity] += 1
                
                print(f"{i:2d}. {dtype} - {severity} Severity")
                print(f"    Location: ({damage['center'][0]}, {damage['center'][1]})")
                print(f"    Size: {damage['length']:.0f} x {damage['width']:.0f} pixels")
                print(f"    Area: {damage['area']:.0f} pixels")
                print()
            
            print("Summary by Type:")
            for dtype, count in type_counts.items():
                print(f"  {dtype}: {count}")
            
            print("\nSummary by Severity:")
            for severity, count in severity_counts.items():
                if count > 0:
                    print(f"  {severity}: {count}")
            
            # Recommendations
            print("\nRecommendations:")
            high_severity = severity_counts['High']
            if high_severity > 0:
                print(f"⚠️  URGENT: {high_severity} high-severity damage(s) need immediate attention!")
            
            medium_severity = severity_counts['Medium']
            if medium_severity > 0:
                print(f"📋 Schedule repair for {medium_severity} medium-severity damage(s)")
            
            low_severity = severity_counts['Low']
            if low_severity > 0:
                print(f"📝 Monitor {low_severity} low-severity damage(s) for changes")
        
        else:
            print("✅ No significant damage detected in this building image.")
        
        print("="*60)


def main():
    """Main function to run building analysis"""
    print("Building Crack Detection System")
    print("="*50)
    
    # Initialize the detection system
    detector = CrackDetectionSystem()
    
    # Look for building.jpg in current directory
    image_path = "building.jpg"
    
    if not os.path.exists(image_path):
        print(f"Looking for '{image_path}' in current directory...")
        print(f"Current directory: {os.getcwd()}")
        print("\nFiles in current directory:")
        for file in os.listdir('.'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                print(f"  📸 {file}")
        
        print(f"\n❌ '{image_path}' not found!")
        print("Please make sure 'building.jpg' is in the same folder as this script.")
        return
    
    # Process the building image
    try:
        result = detector.process_image(image_path)
        
        if result:
            # Save detailed report
            report_path = os.path.join(detector.config['output_dir'], 'building_analysis_report.txt')
            with open(report_path, 'w') as f:
                f.write(f"Building Damage Analysis Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Image: {result['image_name']}\n")
                f.write(f"Total Damages: {result['total_damages']}\n\n")
                
                for i, damage in enumerate(result['damages'], 1):
                    f.write(f"{i}. {damage['type']} - {damage['severity']} Severity\n")
                    f.write(f"   Location: {damage['center']}\n")
                    f.write(f"   Dimensions: {damage['length']:.0f} x {damage['width']:.0f}\n")
                    f.write(f"   Area: {damage['area']:.0f} pixels\n\n")
            
            print(f"\n📄 Detailed report saved to: {report_path}")
            print("🎉 Analysis complete! Check the results folder for all outputs.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Please check that the image file is valid and not corrupted.")


if __name__ == "__main__":
    main()