# CyberCodeMini — Google Colab Training Guide

This directory contains the self-contained Google Colab notebook for fine-tuning **CyberCodeMini v0.3.0** (`Qwen/Qwen2.5-Coder-1.5B-Instruct`) using free/low-cost GPU resources (Option 3).

---

## Notebook Overview

- **File**: [`CyberCodeMini_Colab_Training.ipynb`](file:///c:/Users/user/OneDrive/Documents/Desktop/cybercode/notebooks/CyberCodeMini_Colab_Training.ipynb)
- **Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Method**: QLoRA (4-bit NF4 quantization + LoRA rank `r=16`, alpha `32`)
- **Dataset**: Frozen `v0.3.0` (900 train candidates, 90 validation candidates, 200 evaluation items)
- **SHA-256 Hash**: `c58c523cd8a7b6316054ca7289f98a427e4d95611ab4a27b024033c304314df5`
- **Hardware Requirement**: T4 GPU (15 GB VRAM) or A100 GPU on Google Colab (Free or Colab Pro)
- **Estimated Execution Time**: ~25–35 minutes on T4 (~8 minutes on A100)

---

## How to Run on Google Colab

1. **Upload Notebook**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Click **File > Upload Notebook** and select [`notebooks/CyberCodeMini_Colab_Training.ipynb`](file:///c:/Users/user/OneDrive/Documents/Desktop/cybercode/notebooks/CyberCodeMini_Colab_Training.ipynb) from this repository.

2. **Enable GPU Accelerator**:
   - Click **Runtime > Change runtime type**.
   - Select **T4 GPU** (or A100 GPU) under Hardware accelerator.

3. **Execute Cells**:
   - Click **Runtime > Run all** (or execute cells sequentially `Shift + Enter`).
   - The notebook will clone the project repository `https://github.com/griffin6569/cybercode_v1.git`, verify dataset SHA-256 integrity, train the model, test inference, and download the zipped LoRA adapter (`cybercodemini_lora_v0.3.0_adapter.zip`).

4. **Save Trained Adapter**:
   - The final adapter weights will automatically download to your local machine as a `.zip` archive.
