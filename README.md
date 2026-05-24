# **TRELLIS.2-Text-to-3D-FA2**

TRELLIS.2-Text-to-3D-FA2 is an advanced, experimental 3D asset generation suite that couples high-speed text-to-image synthesis with structured 3D geometry reasoning. By linking Alibaba's rapid `Tongyi-MAI/Z-Image-Turbo` pipeline with Microsoft's state-of-the-art `microsoft/TRELLIS.2-4B` foundational 3D asset model, this application delivers an end-to-end framework capable of converting raw text descriptions or pre-existing images into textured 3D assets. The backend utilizes Flash Attention 2 alongside a cascade of specialized sparse structures, shape vaes, and material sampling solvers to yield refined spatial models. Wrapped in a high-fidelity web workspace featuring an Orange Red theme, it hosts an embedded spatial Rerun viewport that interactive maps point grids, camera positions, and texture structures within seconds of execution.

> [!IMPORTANT]
Device Support and Testing (Blackwell Architecture): NVIDIA RTX PRO 6000 Blackwell Server Edition, Kernel (Memory-Efficient): FlashAttention 2

<img width="1771" height="1546" alt="screencapture-huggingface-co-spaces-prithivMLmods-TRELLIS-2-Text-to-3D-2026-05-14-20_02_24" src="https://github.com/user-attachments/assets/031ef49d-27f6-471d-984f-a427ed8887a5" />

### **Key Features**

* **Unified Generation Pipelines:** Supports both Text-to-Image-3D (via Z-Image Turbo integration) and high-fidelity direct Image-to-3D conversion workflows.
* **Cascaded Multi-Stage Samplers:** Grants granular control over separate algorithmic generation layers, including Stage 1 Sparse Structures, Stage 2 Shapes, and Stage 3 Surface PBR Materials.
* **Automated Background Removal:** Native integration with BRIA RMBG 2.0 automatically isolates, masks, and centers subject boundaries out of raw inputs prior to 3D voxel baking.
* **Interactive Rerun 3D Canvas:** Houses an interactive Rerun viewport with coordinate axis helpers allowing real-time viewport orbit, bounding box inspection, and spatial measurement.
* **Standardized Asset Baking:** Outputs fully cross-compatible `.glb` file geometry with adaptive triangle decimations (up to 500,000 target faces) and selectable baking texture resolutions (up to 4096px).

### **Repository Structure**

```text
├── assets/
│   ├── app/
│   ├── example_image/
│   └── hdri/
├── example-images/
├── trellis2/
│   ├── datasets/
│   ├── models/
│   ├── modules/
│   ├── pipelines/
│   ├── renderers/
│   ├── representations/
│   ├── trainers/
│   └── utils/
├── app.py
├── autotune_cache.json
├── LICENSE.txt
├── pre-requirements.txt
├── README.md
└── requirements.txt

```

### **Installation and Requirements**

To initialize the TRELLIS.2 workspace locally, configure a Python environment equipped with specialized pre-compiled wheels for CUDA 13.0 and PyTorch 2.11.0.

**1. Install Pre-requirements**
Ensure your project environment updates pip to handle modern package wrappers:

```bash
pip install pip>=23.0.0

```

**2. Install Core Requirements**
Install the necessary machine learning, geometric rendering, and backend packages. Place these dependencies inside a `requirements.txt` file and run `pip install -r requirements.txt`.

```text
--extra-index-url https://download.pytorch.org/whl/cu130
-f https://whl.natten.org

torch==2.11.0
torchvision==0.26.0
triton==3.6.0
pillow==12.0.0
imageio==2.37.2
imageio-ffmpeg==0.6.0
tqdm==4.67.1
easydict==1.13
opencv-python-headless==4.12.0.88
trimesh==4.10.1
transformers==4.57.3
zstandard==0.25.0
kornia==0.8.2
timm==1.0.22
diffusers==0.37.1
accelerate==1.13.0
gradio
einops==0.8.2
plyfile==1.1.3
git+https://github.com/microsoft/MoGe.git
spaces

natten==0.21.6+torch2110cu130
https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
https://github.com/LDYang694/Storages/releases/download/rtxpro6000/flex_gemm-1.0.0%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl
https://github.com/LDYang694/Storages/releases/download/rtxpro6000/nvdiffrast-0.4.0%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl
https://github.com/LDYang694/Storages/releases/download/rtxpro6000/nvdiffrec_render-0.0.0%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl
https://github.com/LDYang694/Storages/releases/download/rtxpro6000/cumesh-0.0.1%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl
https://github.com/LDYang694/Storages/releases/download/rtxpro6000/o_voxel-0.0.1%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl

omegaconf
termcolor
icecream
pyserde
gradio==6.14.0
rerun-sdk
gradio_rerun
scipy
jax
jaxtyping
monopriors
braceexpand

```

---

### **Usage**

Once your infrastructure configuration and compiled dependencies are validated, execute the application by invoking the main Python runtime file:

```bash
python app.py

```

The server initialization phase prepares target autotune caches and brings up both the image diffusion pipeline and the 4B parameter TRELLIS architecture into device VRAM. When complete, log in through your preferred browser at the local loopback address provided in your terminal (typically `http://127.0.0.1:7860/`).

* **To create from text:** Write a clear structural prompt in the text tab (e.g., *"A vintage car, low poly style"*), click **1. Generate Image**, inspect the result, and pass it forward by clicking **2. Generate 3D**.
* **To create from an image:** Toggle to the Image-to-3D module, upload a reference asset with a clean background mask, and click **2. Generate 3D**. The final asset will render inside the live Rerun interactive panel and become available for download as a standardized `.glb` model file.

### **License and Source**

* **License:** [Apache License Version 2.0](https://www.google.com/search?q=https://github.com/PROTIVSAKTHIUR/TRELLIS.2-Text-to-3D-FA2/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/TRELLIS.2-Text-to-3D-FA2](https://github.com/PRITHIVSAKTHIUR/TRELLIS.2-Text-to-3D-FA2)
