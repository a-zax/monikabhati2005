import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QPushButton, QFileDialog, QTextEdit, QProgressBar, 
                               QFrame, QSplitter, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon, QAction
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cognitive_model import CognitiveReportGenerator
from models.dataset import get_transforms

# --- Styling ---
DARK_THEME = """
QMainWindow {
    background-color: #1e1e1e;
    color: #ffffff;
}
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 14px;
}
QPushButton {
    background-color: #0d47a1;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1565c0;
}
QPushButton:pressed {
    background-color: #0d47a1;
}
QPushButton:disabled {
    background-color: #424242;
    color: #757575;
}
QTextEdit {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 5px;
    padding: 10px;
    color: #e0e0e0;
    font-family: 'Consolas', 'Monaco', monospace;
}
QLabel {
    color: #e0e0e0;
}
QProgressBar {
    border: 1px solid #3d3d3d;
    border-radius: 5px;
    text-align: center;
    background-color: #2d2d2d;
}
QProgressBar::chunk {
    background-color: #0d47a1;
    border-radius: 4px;
}
QFrame#Separator {
    background-color: #3d3d3d;
    color: #3d3d3d;
}
"""

class ModelLoaderThread(QThread):
    finished = Signal(object, object, object, object) # model, enc_tokenizer, dec_tokenizer, device
    error = Signal(str)

    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def run(self):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            
            # Assumptions about model config (should match training)
            model = CognitiveReportGenerator(
                visual_encoder='vit_base_patch16_224',
                text_encoder_name='distilbert-base-uncased',
                decoder_name='distilgpt2',
                num_diseases=14,
                hidden_dim=512
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            enc_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            dec_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            if dec_tokenizer.pad_token is None:
                dec_tokenizer.pad_token = dec_tokenizer.eos_token
                
            self.finished.emit(model, enc_tokenizer, dec_tokenizer, device)
        except Exception as e:
            self.error.emit(str(e))

class InferenceThread(QThread):
    finished = Signal(str, dict)
    error = Signal(str)

    def __init__(self, model, image_path, enc_tokenizer, dec_tokenizer, device):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.device = device

    def run(self):
        try:
             # Load Image
            image = Image.open(self.image_path).convert('RGB')
            image_np = np.array(image)
            transform = get_transforms(is_train=False)
            augmented = transform(image=image_np)
            image_tensor = augmented['image'].unsqueeze(0).to(self.device)

            # Context
            indication_text = "Clinical indication: Routine check."
            indication = self.enc_tokenizer(
                indication_text, max_length=64, padding='max_length', truncation=True, return_tensors='pt'
            )
            indication_ids = indication['input_ids'].to(self.device)
            indication_mask = indication['attention_mask'].to(self.device)

            with torch.no_grad():
                generated_ids, disease_probs = self.model.generate(
                    image_tensor, indication_ids, indication_mask, max_length=256
                )
                
                report = self.dec_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                diseases = [
                    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
                    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
                ]
                disease_dict = {d: p.item() for d, p in zip(diseases, disease_probs[0])}
                
            self.finished.emit(report, disease_dict)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cognitive Radiology Assistant (BrainDead 2K26)")
        self.resize(1200, 800)
        self.setStyleSheet(DARK_THEME)
        
        # State
        self.model = None
        self.device = None
        self.current_image_path = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("Cognitive Radiology Report Generator")
        title_label.setFont(QFont('Segoe UI', 18, QFont.Bold))
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        self.load_model_btn = QPushButton("Load Model Checkpoint")
        self.load_model_btn.clicked.connect(self.load_model_dialog)
        header_layout.addWidget(self.load_model_btn)
        
        main_layout.addLayout(header_layout)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setObjectName("Separator")
        main_layout.addWidget(line)

        # Splitter for Content
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left Panel: Image
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        self.image_label = QLabel("No Image Selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #252525; border-radius: 10px; border: 2px dashed #3d3d3d;")
        self.image_label.setMinimumSize(400, 400)
        left_layout.addWidget(self.image_label)
        
        self.load_image_btn = QPushButton("Select X-Ray Image")
        self.load_image_btn.clicked.connect(self.load_image_dialog)
        left_layout.addWidget(self.load_image_btn)
        
        splitter.addWidget(left_panel)
        
        # Right Panel: Report & Findings
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        # Report Section
        right_layout.addWidget(QLabel("Generated Clinical Report:"))
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("Report will appear here...")
        right_layout.addWidget(self.report_text)
        
        # Findings Section
        right_layout.addWidget(QLabel("Detected Findings (>50% Probability):"))
        self.findings_container = QWidget()
        self.findings_layout = QVBoxLayout(self.findings_container)
        self.findings_layout.setAlignment(Qt.AlignTop)
        right_layout.addWidget(self.findings_container)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 700])
        
        # Footer Action
        self.generate_btn = QPushButton("GENERATE REPORT")
        self.generate_btn.setFixedHeight(50)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #00c853; 
                font-size: 16px;
            }
            QPushButton:hover { background-color: #00e676; }
            QPushButton:disabled { background-color: #424242; }
        """)
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.start_generation)
        main_layout.addWidget(self.generate_btn)
        
        # Status Bar
        self.status_bar = QLabel("Please load a model to begin.")
        self.status_bar.setStyleSheet("color: #757575; font-style: italic;")
        main_layout.addWidget(self.status_bar)

    def load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", "", "PyTorch Models (*.pth)")
        if path:
            self.load_model_btn.setEnabled(False)
            self.status_bar.setText(f"Loading model from {os.path.basename(path)}... This may take a moment.")
            self.loader_thread = ModelLoaderThread(path)
            self.loader_thread.finished.connect(self.on_model_loaded)
            self.loader_thread.error.connect(self.on_model_error)
            self.loader_thread.start()

    def on_model_loaded(self, model, enc_tok, dec_tok, device):
        self.model = model
        self.enc_tokenizer = enc_tok
        self.dec_tokenizer = dec_tok
        self.device = device
        self.load_model_btn.setText("Model Loaded ✓")
        self.status_bar.setText("Model loaded successfully. Ready to generate.")
        self.check_ready()

    def on_model_error(self, err):
        QMessageBox.critical(self, "Error", f"Failed to load model:\n{err}")
        self.load_model_btn.setEnabled(True)
        self.status_bar.setText("Error loading model.")

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.current_image_path = path
            self.display_image(path)
            self.check_ready()

    def display_image(self, path):
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.status_bar.setText(f"Image selected: {os.path.basename(path)}")

    def check_ready(self):
        if self.model and self.current_image_path:
            self.generate_btn.setEnabled(True)

    def start_generation(self):
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("GENERATING...")
        self.report_text.clear()
        self.clear_findings()
        self.status_bar.setText("Analyzing image and generating report...")
        
        self.inf_thread = InferenceThread(
            self.model, self.current_image_path, self.enc_tokenizer, self.dec_tokenizer, self.device
        )
        self.inf_thread.finished.connect(self.on_generation_finished)
        self.inf_thread.error.connect(self.on_generation_error)
        self.inf_thread.start()

    def on_generation_finished(self, report, disease_probs):
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("GENERATE REPORT")
        self.status_bar.setText("Generation complete.")
        
        # Typewriter effect (simulated by just setting text for now)
        self.report_text.setText(report)
        
        # Display findings
        self.display_findings(disease_probs)

    def on_generation_error(self, err):
        QMessageBox.critical(self, "Error", f"Inference failed:\n{err}")
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("GENERATE REPORT")
        self.status_bar.setText("Generation failed.")

    def clear_findings(self):
        # Remove all widgets from findings layout
        while self.findings_layout.count():
            child = self.findings_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def display_findings(self, probs):
        self.clear_findings()
        
        has_findings = False
        for disease, prob in probs.items():
            if prob > 0.5:
                has_findings = True
                
                container = QWidget()
                layout = QHBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
                
                label = QLabel(f"{disease}")
                label.setFixedWidth(200)
                
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(int(prob * 100))
                bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #3d3d3d;
                        border-radius: 4px;
                        text-align: right;
                        padding-right: 5px;
                        background: #2d2d2d;
                    }
                    QProgressBar::chunk {
                        background-color: #ff5252;
                    }
                """)
                if prob < 0.7:
                    bar.setStyleSheet(bar.styleSheet().replace("#ff5252", "#ffa726")) # Orange for medium risk
                
                layout.addWidget(label)
                layout.addWidget(bar)
                
                self.findings_layout.addWidget(container)
        
        if not has_findings:
            label = QLabel("No significant abnormalities detected.")
            label.setStyleSheet("color: #69f0ae; font-style: italic;")
            self.findings_layout.addWidget(label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
