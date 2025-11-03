# transformer-slam
A hands-on 12-week Robotics and AI integration mini project | Pytorch -> LoFTR style matching -> CUDA

# Transformer-SLAM 🚀  
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)  
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)](https://developer.nvidia.com/cuda-toolkit)  
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)  

> A 12-week structured roadmap to build a **Transformer-based SLAM pipeline** using PyTorch and CUDA — from perception to planning.

---

## 🧩 Overview
This repository provides a **modular, week-by-week learning plan** to go from PyTorch fundamentals to a fully working **Transformer-based SLAM** system.  
The end goal: **build, train, and extend** a Transformer model that performs both **visual SLAM** and **trajectory planning**.

### Highlights
- 🔥 End-to-end Transformer SLAM architecture  
- ⚙️ CUDA optimization & performance profiling  
- 🧭 Decision Transformer / Diffusion Policy extension  
- 📊 Portfolio-ready results & documentation  

---

## 📁 Repository Structure

```
Transformer-SLAM/
│
├── phase1_pytorch_basics/
│   ├── train_vit_toy.py
│   └── attention_visualization.ipynb
│
├── phase2_feature_matching/
│   └── loftr_simplified.py
│
├── phase3_pose_estimation_cuda/
│   └── pose_from_correspondences.py
│
├── phase4_planning_extension/
│   └── transformer_planner.py
│
├── data/                     # KITTI / TUM-RGBD data (not included)
├── outputs/                  # Logs, plots, and trained weights
├── environment.yml           # Conda environment
├── README.md
└── LICENSE
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Transformer-SLAM.git
cd Transformer-SLAM
```

### 2. Create a Conda environment
```bash
conda env create -f environment.yml
conda activate transformer_slam
```

### 3. Install extra dependencies
```bash
pip install torch torchvision kornia opencv-python matplotlib evo
```

### 4. Verify CUDA setup
```python
import torch
print(torch.cuda.is_available())
```

---

## 🧠 Learning Roadmap

### **Phase 1 – PyTorch Foundations (Weeks 1–3)**
- Linear regression & CNNs from scratch  
- Custom Dataset & DataLoader  
- ViT (toy version) + attention visualization  

### **Phase 2 – Transformer Feature Matching (Weeks 4–6)**
- LoFTR-style feature matching  
- KITTI/TUM-RGBD dataset training  
- Kornia feature matching + AMP training  

### **Phase 3 – Pose Estimation & CUDA (Weeks 7–9)**
- Essential matrix estimation using OpenCV  
- GPU profiling with Nsight Systems  
- Custom CUDA kernel  

### **Phase 4 – Planning Extension (Weeks 10–12)**
- Decision Transformer or Diffusion Planner  
- Sequence modeling on robot state data  
- Comparative visualization of results  

---

## 🧰 Tools & Dependencies

| Category | Tools |
|-----------|-------|
| **Deep Learning** | PyTorch, torchvision, Kornia |
| **CUDA & Profiling** | torch.profiler, Nsight Systems |
| **Vision & Geometry** | OpenCV, NumPy |
| **Datasets** | KITTI, TUM-RGBD |
| **Visualization** | Matplotlib, evo |
| **Environment** | Ubuntu + Conda (GPU RTX 3060+ recommended) |

---

## 📊 Deliverables
- ✅ Transformer-based SLAM model  
- ✅ CUDA kernel + profiling report  
- ✅ Planning module (Decision Transformer / Diffusion Policy)  
- ✅ Visualized SLAM & planning comparisons  

---

## 🧭 References
- [PyTorch Tutorials](https://pytorch.org/tutorials/)  
- [LoFTR: Detector-Free Local Feature Matching](https://arxiv.org/abs/2104.00680)  
- [Decision Transformer](https://arxiv.org/abs/2106.01345)  
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)  
- [OpenCV Pose Estimation Docs](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

---

## 🪪 License
This project is licensed under the [MIT License](LICENSE).  
Feel free to fork, modify, and share it for educational or research purposes.

---

### 💬 Author
**Sawan Saurabh** – Robotics Software Engineer  
🌐 [LinkedIn](https://linkedin.com/in/sawan-saurabh) | [GitHub](https://github.com/sawansaurabh-offcl)

---

> _"From pixels to plans — building the bridge between perception and decision."_  
