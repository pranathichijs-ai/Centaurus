Robust Detection of AI-Generated Images Under Real-World Transformations
Project Overview
<!-- 2-4 sentences: what does your solution do, and why does robustness to real-world transformations matter for AIGC detection? Tie back to the misinformation / impersonation / fraud angle from the problem statement. -->
Problem Addressed
<!-- Restate the problem in your own words. Emphasize the "robustness after compression/cropping/reposting" angle, not just "detect AI images." -->
Approach
<!-- High-level description of your technical approach: - Model architecture (and why, given the <2B parameter constraint) - Training strategy (including augmentation for robustness) - Any frequency-domain / feature engineering used -->
Datasets Used
Dataset	Purpose	Notes
CIFAKE	e.g. baseline training	
SID_Set		
WildFake		
Development Tools
<!-- e.g. VSCode, Colab, Jupyter -->
Models / APIs / Libraries
<!-- e.g. PyTorch, Hugging Face Transformers, scikit-learn, albumentations -->
Setup & Installation
bash
git clone <repo-url>
cd aigc-detection
pip install -r requirements.txt
Steps to Reproduce Results
<!-- Data download / preprocessing step -->
<!-- Training command -->
<!-- Evaluation command -->
Running Inference
bash
python eval/infer.py --input_dir <path_to_images> --output results.json

Output format:

json
[
  {"image_path": "path/to/img1.jpg", "pred": 0.87},
  {"image_path": "path/to/img2.jpg", "pred": 0.12}
]
Robustness Evaluation Summary
<!-- Insert or link to the robustness comparison table (clean vs. transformed). -->
Error Analysis
<!-- Representative false positives, false negatives, and trade-off discussion. -->
Limitations & Future Work
<!-- Brief reflection: what would you improve given more time? -->
Team Contributions
Member	Contribution
	
	
	
	
Demo Video
<!-- YouTube link (public) -->
