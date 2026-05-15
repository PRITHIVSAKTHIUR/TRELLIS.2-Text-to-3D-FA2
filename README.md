# **[TRELLIS.2-Text-to-3D-FA2](https://huggingface.co/spaces/prithivMLmods/TRELLIS.2-Text-to-3D)**

A Gradio with Rerun Embedded demonstration for Microsoft’s TRELLIS.2-4B model with integrated Rerun visualization. It converts text prompts or uploaded images into high-quality, textured 3D assets (GLB) through a two-stage workflow: Text to Image (Z-Image-Turbo) → Image to 3D (TRELLIS.2). The demo features interactive 3D viewing powered by the Rerun SDK, with proper coordinate system setup, axes helpers, and downloadable GLB files.

## Features

- **Text-to-Image-to-3D**: Generate base images from prompts using Z-Image-Turbo, then lift to 3D.
- **Direct Image-to-3D**: Upload RGBA/PNG images; auto-preprocesses with background removal (BRIA-RMBG-2.0) and cropping.
- **Rerun 3D Viewer**: Interactive visualization with correct RIGHT_HAND_Y_UP coordinates, colored axes (X=red, Y=green, Z=blue), and clean 3D view blueprint.
- **Advanced Controls**: Resolutions (512/1024/1536), detailed sampler settings for sparse structure, shape, and material stages, face decimation, texture size.
- **Robust Export**: GLB with PNG textures (extension_webp=False for compatibility); fallback remeshing if high-quality fails.
- **Session Management**: Per-user temp directories; auto-cleanup on unload.
- **Custom Theme**: OrangeRedTheme with responsive layout.
- **Rich Examples**: 70+ image inputs and 60+ text prompts (cats, planes, cars, furniture, etc.).

---

<img width="1771" height="1546" alt="screencapture-huggingface-co-spaces-prithivMLmods-TRELLIS-2-Text-to-3D-2026-05-14-20_02_24" src="https://github.com/user-attachments/assets/031ef49d-27f6-471d-984f-a427ed8887a5" />

---
