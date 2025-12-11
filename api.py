#!/usr/bin/env python3
"""
Flask REST API for PSO brain tumor segmentation.
Provides endpoints for image processing and batch operations.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
import io
import base64
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional

from preprocessing import preprocess_image, load_image
from pso_segmentation import pso_threshold
from metrics import compute_all_metrics
from results_exporter import generate_summary_statistics
import config

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend


def image_to_base64(img: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    pil_img = Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'PSO Segmentation API is running'
    })


@app.route('/api/process', methods=['POST'])
def process_image():
    """
    Process a single image.
    
    Expected JSON:
    {
        "image": "base64_encoded_image",
        "mask": "base64_encoded_mask" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'Image is required'}), 400
        
        # Decode image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image_array = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        
        if image_array is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Preprocess
        img_processed = preprocess_image(image_array)
        
        result = {
            'processed_image': image_to_base64((img_processed * 255).astype(np.uint8)),
            'has_mask': False
        }
        
        # Process mask if provided
        if 'mask' in data and data['mask']:
            mask_data = data['mask'].split(',')[1] if ',' in data['mask'] else data['mask']
            mask_bytes = base64.b64decode(mask_data)
            mask_array = np.frombuffer(mask_bytes, np.uint8)
            mask_array = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            
            if mask_array is not None:
                # Resize mask
                mask_resized = cv2.resize(
                    mask_array,
                    (image_array.shape[1], image_array.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                mask_binary = (mask_resized > 0).astype(np.uint8)
                
                # Process with PSO
                if mask_binary.sum() > 0:
                    threshold = pso_threshold(img_processed, mask_binary)
                    pred = (img_processed > threshold).astype(np.uint8)
                    metrics = compute_all_metrics(mask_binary, pred)
                    
                    result.update({
                        'has_mask': True,
                        'metrics': {
                            'dice': float(metrics[0]),
                            'iou': float(metrics[1]),
                            'precision': float(metrics[2]),
                            'recall': float(metrics[3])
                        },
                        'threshold': float(threshold),
                        'prediction': image_to_base64(pred * 255),
                        'mask': image_to_base64(mask_binary * 255)
                    })
                else:
                    # Healthy slice
                    pred = np.zeros_like(mask_binary, dtype=np.uint8)
                    metrics = compute_all_metrics(mask_binary, pred)
                    result.update({
                        'has_mask': True,
                        'metrics': {
                            'dice': float(metrics[0]),
                            'iou': float(metrics[1]),
                            'precision': float(metrics[2]),
                            'recall': float(metrics[3])
                        },
                        'threshold': None,
                        'prediction': image_to_base64(pred),
                        'mask': image_to_base64(mask_binary * 255),
                        'is_healthy': True
                    })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch', methods=['POST'])
def process_batch():
    """
    Process multiple images in batch.
    
    Expected JSON:
    {
        "images": [
            {"image": "base64", "mask": "base64"},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'images' not in data:
            return jsonify({'error': 'Images array is required'}), 400
        
        results = []
        tumour_metrics = []
        healthy_metrics = []
        tumour_thresholds = []
        
        for item in data['images']:
            if 'image' not in item:
                continue
            
            # Decode image
            image_data = item['image'].split(',')[1] if ',' in item['image'] else item['image']
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, np.uint8)
            image_array = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            
            if image_array is None:
                continue
            
            img_processed = preprocess_image(image_array)
            
            result_item = {'processed': True}
            
            if 'mask' in item and item['mask']:
                mask_data = item['mask'].split(',')[1] if ',' in item['mask'] else item['mask']
                mask_bytes = base64.b64decode(mask_data)
                mask_array = np.frombuffer(mask_bytes, np.uint8)
                mask_array = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
                
                if mask_array is not None:
                    mask_resized = cv2.resize(
                        mask_array,
                        (image_array.shape[1], image_array.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask_binary = (mask_resized > 0).astype(np.uint8)
                    
                    if mask_binary.sum() > 0:
                        threshold = pso_threshold(img_processed, mask_binary)
                        pred = (img_processed > threshold).astype(np.uint8)
                        metrics = compute_all_metrics(mask_binary, pred)
                        
                        tumour_metrics.append(metrics)
                        tumour_thresholds.append(threshold)
                        
                        result_item.update({
                            'metrics': {
                                'dice': float(metrics[0]),
                                'iou': float(metrics[1]),
                                'precision': float(metrics[2]),
                                'recall': float(metrics[3])
                            },
                            'threshold': float(threshold),
                            'is_tumour': True
                        })
                    else:
                        pred = np.zeros_like(mask_binary, dtype=np.uint8)
                        metrics = compute_all_metrics(mask_binary, pred)
                        healthy_metrics.append(metrics)
                        
                        result_item.update({
                            'metrics': {
                                'dice': float(metrics[0]),
                                'iou': float(metrics[1]),
                                'precision': float(metrics[2]),
                                'recall': float(metrics[3])
                            },
                            'is_healthy': True
                        })
            
            results.append(result_item)
        
        # Generate summary statistics
        summary = generate_summary_statistics(
            tumour_metrics, healthy_metrics, tumour_thresholds
        )
        
        return jsonify({
            'results': results,
            'summary': summary,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    return jsonify({
        'pso_particles': config.PSO_N_PARTICLES,
        'pso_iterations': config.PSO_ITERATIONS,
        'pso_options': config.PSO_OPTIONS,
        'pso_bounds': config.PSO_BOUNDS
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
