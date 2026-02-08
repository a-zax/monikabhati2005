#!/usr/bin/env python3
"""
Inference script for Cognitive Report Generator
"""

import torch
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import textwrap

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cognitive_model import CognitiveReportGenerator
from models.dataset import get_transforms

def generate_report(model, image_path, enc_tokenizer, dec_tokenizer, device, max_length=256):
    model.eval()
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        transform = get_transforms(is_train=False)
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(device) # [1, 3, 224, 224]
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    # Dummy indication (clinical context)
    # in a real app, user would provide this
    indication_text = "Clinical indication: Chest pain."
    indication = enc_tokenizer(
        indication_text,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    indication_ids = indication['input_ids'].to(device)
    indication_mask = indication['attention_mask'].to(device)
    
    with torch.no_grad():
        generated_ids, disease_probs = model.generate(
            image_tensor,
            indication_ids, 
            indication_mask,
            max_length=max_length
        )
        
        report = dec_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Format disease probabilities
        diseases = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
            "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
        ]
        disease_dict = {d: p.item() for d, p in zip(diseases, disease_probs[0])}
        
    return report, disease_dict

def visualize_result(image_path, report, disease_probs):
    """
    Creates and saves a visualization of the result.
    """
    try:
        plt.figure(figsize=(12, 6))
        
        # 1. Image
        img = Image.open(image_path).convert('RGB')
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title("Input X-Ray")
        
        # 2. Report Text
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.title("Generated Report")
        
        wrapper = textwrap.TextWrapper(width=40)
        wrapped_text = wrapper.fill(report)
        
        # Add disease findings
        findings = {d: p for d, p in disease_probs.items() if p > 0.5}
        if findings:
            wrapped_text += "\n\nDetected Findings:\n" + "\n".join([f"{d}: {p:.2f}" for d, p in findings.items()])
            
        plt.text(0.1, 0.9, wrapped_text, fontsize=12, va='top', family='monospace')
        
        output_path = "demo_result.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visual result saved to {output_path}")
        print("You can view this image in the Kaggle output section or download it.")
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = CognitiveReportGenerator(
        visual_encoder=args.visual_encoder,
        text_encoder_name=args.text_encoder,
        decoder_name=args.decoder,
        num_diseases=14,
        hidden_dim=512
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    enc_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    dec_tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    
    if dec_tokenizer.pad_token is None:
        dec_tokenizer.pad_token = dec_tokenizer.eos_token
        
    # Generate
    print(f"Generating report for {args.image}...")
    report, disease_probs = generate_report(model, args.image, enc_tokenizer, dec_tokenizer, device)
    
    if report is None:
        print("Failed to generate report.")
        return

    print("\n" + "="*50)
    print("GENERATED REPORT")
    print("="*50)
    print(report)
    print("="*50)
    
    print(f"\nDisease Probabilities:")
    for d, p in disease_probs.items():
        if p > 0.5:
             print(f"{d}: {p:.4f} (*)")
        else:
             print(f"{d}: {p:.4f}")

    # Create visualization
    visualize_result(args.image, report, disease_probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, default=None, help="Path to image. If None, picks a random one.")
    parser.add_argument('--dataset_root', type=str, default='/kaggle/input', help="Root to search if no image provided")
    parser.add_argument('--visual_encoder', type=str, default='vit_base_patch16_224')
    parser.add_argument('--text_encoder', type=str, default='distilbert-base-uncased')
    parser.add_argument('--decoder', type=str, default='distilgpt2')
    
    args = parser.parse_args()
    
    if args.image is None:
        print("No image provided. Searching for a sample image...")
        import os
        for root, dirs, files in os.walk(args.dataset_root):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    args.image = os.path.join(root, file)
                    print(f"Index found sample: {args.image}")
                    break
            if args.image: break
            
        if not args.image:
            print("Error: Could not find any images to test.")
            sys.exit(1)
            
    main(args)
