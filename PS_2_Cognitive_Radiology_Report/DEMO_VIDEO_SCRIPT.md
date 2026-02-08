# Demo Video Script: Cognitive Radiology Assistant
**Duration:** 2 minutes  
**Purpose:** Demonstrate medical report generation from chest X-ray using PRO-FA, MIX-MLP, and RCTA modules

---

## Pre-Recording Checklist
- [ ] GUI application tested and running smoothly
- [ ] Sample X-ray image prepared
- [ ] Model checkpoint loaded successfully
- [ ] Screen recording software ready (OBS/Xbox Game Bar)
- [ ] Microphone tested (optional, can add voiceover later)

---

## Script Timeline

### [0:00-0:15] Introduction & Interface Overview
**Narration:**
> "Welcome to the Cognitive Radiology Assistant—an AI system that generates comprehensive diagnostic reports from chest X-rays. The system implements three core modules: PRO-FA for hierarchical visual encoding, MIX-MLP for multi-disease classification, and RCTA for cognitive triangular attention."

**Actions:**
- Show full GUI interface
- Highlight team name "monikabhati2005" in header
- Show clean, professional layout

---

### [0:15-0:35] Load Neural Weights
**Narration:**
> "First, we load the trained model checkpoint containing our neural network weights."

**Actions:**
1. Click "Load Neural Weights" button
2. File dialog appears
3. Navigate to `checkpoints/` folder
4. Select checkpoint file (e.g., `best_model_epoch_10.pth`)
5. Click Open
6. Show progress bar animation (Neural Ribbon effect)
7. Wait for "Backbone Active ✓" status
8. Status bar shows: "Neural Backbone Active. Ready for input."

---

### [0:35-0:55] LoadChest X-Ray Image
**Narration:**
> "Next, we load a sample chest radiograph for analysis."

**Actions:**
1. Click "Pick X-Ray Source" button
2. File dialog appears
3. Select sample X-ray from `data/sample_images/` or test set
4. Image loads with smooth fade-in animation
5. Image displays in left panel with clean border
6. Status updates: "Scanning Source: [filename]"

---

### [0:55-1:35] Execute Diagnostic Scan
**Narration:**
> "Now we execute the complete diagnostic pipeline. Watch as the three modules work together: PRO-FA extracts visual features, MIX-MLP classifies diseases, and RCTA aligns everything through triangular attention."

**Actions:**
1. Click "EXECUTE COMPLETE SCAN" button
2. Button text changes to "DIAGNOSTIC CORE COGNITION..."
3. **Point out visual feedback:**
   - Heartbeat pulse appears on X-ray (blue pulsating glow)
   - 14-segment disease rail lights up sequentially above
   - Circular progress ring animates on button
   - Progress bar flows across header
4. Wait for completion (~3-5 seconds)
5. Show smooth transitions as animations stop

---

### [1:35-2:00] Review Generated Report
**Narration:**
> "The system has generated a comprehensive diagnostic report. We can see the clinical impression text, along with detected pathologies ranked by confidence. High-risk findings are highlighted in red with pulsing shadows for emphasis."

**Actions:**
1. Show AI Impression text (generated report)
2. Scroll through "Anomaly Analysis Probabilities"
3. Highlight findings above 50% threshold:
   - Point to pathology name (e.g., "Cardiomegaly")
   - Show progress bar with color coding (blue/amber/red)
   - Show confidence percentage (e.g., "78.3%")
4. Point out visual hierarchy:
   - Higher confidence = warmer colors
   - Pulsing shadow on critical findings (>80%)
5. Show "Reset Analysis" button for next scan

---

### [2:00] Closing
**Narration:**
> "This demonstrates all three mandatory modules working seamlessly: hierarchical feature extraction, multi-task classification, and cognitive attention—ready for clinical deployment. Thank you!"

**Actions:**
- Hold on final screen showing complete interface
- Fade to black or show team name

---

## Recording Tips

### Video Settings
- **Resolution:** 1920×1080 (1080p)
- **Frame Rate:** 30 FPS
- **Format:** MP4 (H.264 codec)

### Audio (if recording voiceover)
- Speak clearly and at moderate pace
- Reduce background noise
- Use script as guide, not word-for-word

### Editing (optional)
- Add text overlays for module names:
  * "PRO-FA: Hierarchical Visual Alignment"
  * "MIX-MLP: Disease Classification"
  * "RCTA: Triangular Cognitive Attention"
- Add arrows/highlights during key moments
- Speed up loading times (2x) if needed

---

## Upload Instructions

1. **YouTube (Recommended)**
   - Upload as Unlisted
   - Title: "Cognitive Radiology Assistant - BrainDead 2K26 Demo"
   - Add to video description:
     ```
     Team: monikabhati2005
     Hackathon: BrainDead 2026
     Problem Statement 2: Cognitive Radiology Report Generation

     Modules Demonstrated:
     - PRO-FA (Progressive Feature Alignment)
     - MIX-MLP (Multi-task Knowledge-Enhanced MLP)
     - RCTA (Recursive Cognitive Triangular Attention)

     GitHub: [link]
     Report: [link]
     ```

2. **Google Drive (Alternative)**
   - Upload video file
   - Right-click → Share → Anyone with link can view
   - Copy link

3. **Update submission.txt**
   - Add video link under PS_2 section

---

## Fallback (No Trained Model)

If you don't have a trained checkpoint:

1. Use mock/dummy mode (if implemented)
2. Show GUI interface and explain functionality
3. Use slides/screenshots instead:
   - Slide 1: Architecture diagram
   - Slide 2: Code snippets of each module
   - Slide 3: Expected output format
   - Narrate over slides

---

## Quality Checklist

Before finalizing:
- [ ] Video is clear and legible (1080p)
- [ ] All text on screen is readable
- [ ] Animations are smooth (no lag)
- [ ] Audio is clear (if included)
- [ ] Video length is under 2:30
- [ ] Demonstrates all 3 modules
- [ ] Shows complete inference pipeline
- [ ] Uploaded and link is shareable

---

**Estimated Recording Time:** 15-20 minutes (including retakes)  
**Estimated Editing Time:** 10-15 minutes (if needed)

Good luck! 🎬🚀
