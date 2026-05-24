import os
import shutil
import torch
import numpy as np
from PIL import Image
import tempfile
import uuid
from typing import *
from datetime import datetime
from pathlib import Path

from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

import rerun as rr
# Attempt to import blueprint for advanced view configuration
try:
    import rerun.blueprint as rrb
except ImportError:
    rrb = None
    
from gradio_rerun import Rerun

# --- Theme Configuration ---
colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# --- Environment Setup ---
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "flash_attn_2"
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

import gradio as gr
from gradio_client import Client, handle_file
import spaces
from diffusers import DiffusionPipeline
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

print("Initializing models...")

print("Loading Z-Image-Turbo...")
try:
    z_pipe = DiffusionPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_pipe.to(device)
    print("Z-Image-Turbo loaded.")
except Exception as e:
    print(f"Failed to load Z-Image-Turbo: {e}")
    z_pipe = None

print("Loading TRELLIS.2...")
try:
    trellis_pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    trellis_pipeline.rembg_model = None
    trellis_pipeline.low_vram = False
    trellis_pipeline.cuda()
    print("TRELLIS.2 loaded.")
except Exception as e:
    print(f"Failed to load TRELLIS.2: {e}")
    trellis_pipeline = None

rmbg_client = Client("briaai/BRIA-RMBG-2.0")

def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)

def remove_background(input: Image.Image) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        input = input.convert('RGB')
        input.save(f.name)
        output = rmbg_client.predict(handle_file(f.name), api_name="/image")[0][0]
        output = Image.open(output)
        return output

def preprocess_image(input: Image.Image) -> Image.Image:
    """Preprocess the input image: remove bg, crop, resize."""
    if input is None:
        return None
        
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
    if has_alpha:
        output = input
    else:
        output = remove_background(input)
    
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    if bbox.size == 0:
        return output 
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output

def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

@spaces.GPU(size="xlarge", duration=120)
def generate_txt2img(prompt, progress=gr.Progress(track_tqdm=True)):
    """Generate Image using Z-Image Turbo"""
    if z_pipe is None:
        raise gr.Error("Z-Image-Turbo model failed to load.")
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(42)
    
    progress(0.1, desc="Generating Text-to-Image...")
    try:
        result = z_pipe(
            prompt=prompt,
            negative_prompt=None,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
        )
        return result.images[0]
    except Exception as e:
        raise gr.Error(f"Z-Image Generation failed: {str(e)}")

@spaces.GPU(size="xlarge", duration=120)
def generate_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    decimation_target: int,
    texture_size: int,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_guidance: float,
    shape_rescale: float,
    shape_steps: int,
    shape_rescale_t: float,
    tex_guidance: float,
    tex_rescale: float,
    tex_steps: int,
    tex_rescale_t: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    
    if image is None:
        raise gr.Error("Please provide an input image.")
    
    if trellis_pipeline is None:
        raise gr.Error("TRELLIS model is not loaded.")

    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    progress(0.1, desc="Generating 3D Geometry...")
    try:
        outputs, latents = trellis_pipeline.run(
            image,
            seed=seed,
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_steps,
                "guidance_strength": shape_guidance,
                "guidance_rescale": shape_rescale,
                "rescale_t": shape_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_steps,
                "guidance_strength": tex_guidance,
                "guidance_rescale": tex_rescale,
                "rescale_t": tex_rescale_t,
            },
            pipeline_type={"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[resolution],
            return_latent=True,
        )
        
        # 2. Process Mesh
        progress(0.7, desc="Processing Mesh...")
        mesh = outputs[0]
        # FIX: Aggressive simplification to prevent CUDA errors during post-processing
        mesh.simplify(1000000) 
        
        # 3. Export to GLB
        progress(0.9, desc="Baking Texture & Exporting GLB...")
        
        grid_size = latents[2] 
        
        # Safe export with fallback if high-quality hole filling fails
        try:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=trellis_pipeline.pbr_attr_layout,
                grid_size=grid_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=True, 
                remesh_band=1,
                remesh_project=0,
                use_tqdm=True,
            )
        except RuntimeError as e:
            print(f"Warning: Post-processing failed with remesh=True. Error: {e}")
            print("Retrying with remesh=False (Standard mesh generation)...")
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=trellis_pipeline.pbr_attr_layout,
                grid_size=grid_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=False, 
                remesh_band=1,
                remesh_project=0,
                use_tqdm=True,
            )
        
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%dT%H%M%S")
        glb_path = os.path.join(user_dir, f'trellis_output_{timestamp}.glb')
        
        # FIX: extension_webp=False ensures compatibility with Rerun/Standard Viewers
        glb.export(glb_path, extension_webp=False)
        
        # --- Rerun Visualization Logic ---
        progress(0.95, desc="Creating Rerun Visualization...")
        
        run_id = str(uuid.uuid4())
        
        # Robustly handle different Rerun SDK versions
        rec = None
        if hasattr(rr, "new_recording"):
            rec = rr.new_recording(application_id="TRELLIS-3D-Viewer", recording_id=run_id)
        elif hasattr(rr, "RecordingStream"):
            rec = rr.RecordingStream(application_id="TRELLIS-3D-Viewer", recording_id=run_id)
        else:
            rr.init("TRELLIS-3D-Viewer", recording_id=run_id, spawn=False)
            rec = rr
            
        # 1. Clear State
        rec.log("world", rr.Clear(recursive=True), static=True)

        # 2. Set View Coordinates: RIGHT_HAND_Y_UP
        # This defines +Y as Up, +X as Right, and +Z as "Back" (towards the viewer).
        # This effectively places the default camera in front of the object (at +Z).
        rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        # 3. Add Axes Helpers (Red=X, Green=Y, Blue=Z)
        try:
            rec.log("world/axes/x", rr.Arrows3D(vectors=[[0.5, 0, 0]], colors=[[255, 0, 0]]), static=True)
            rec.log("world/axes/y", rr.Arrows3D(vectors=[[0, 0.5, 0]], colors=[[0, 255, 0]]), static=True)
            rec.log("world/axes/z", rr.Arrows3D(vectors=[[0, 0, 0.5]], colors=[[0, 0, 255]]), static=True)
        except Exception:
            pass

        # 4. Log the 3D Model
        rec.log("world/model", rr.Asset3D(path=glb_path), static=True)
        
        # 5. Send Blueprint (if supported) to force a clean 3D view
        if rrb is not None:
            try:
                # Create a simple blueprint with a 3D view of "world"
                blueprint = rrb.Blueprint(
                    rrb.Spatial3DView(
                        origin="/world",
                        name="3D View",
                    ),
                    collapse_panels=True,
                )
                rec.send_blueprint(blueprint)
            except Exception as e:
                print(f"Blueprint creation failed (non-fatal): {e}")

        # Save the Rerun recording (.rrd)
        rrd_path = os.path.join(user_dir, f'trellis_output_{timestamp}.rrd')
        rec.save(rrd_path)

        # Clean up
        torch.cuda.empty_cache()
        return rrd_path, glb_path
        
    except Exception as e:
        torch.cuda.empty_cache()
        raise gr.Error(f"Generation failed: {str(e)}")

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
#main-title h1 {font-size: 2.4em !important;}
"""

if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    with gr.Blocks() as demo:
        gr.Markdown("# **TRELLIS.2 (Text-to-3D)**", elem_id="main-title")
        gr.Markdown("""
        **Workflow:**
        Generate a 3D asset directly by converting Text-to-Image → 3D or Image-to-3D, powered by [TRELLIS.2](https://huggingface.co/microsoft/TRELLIS.2-4B) and [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).
        """)

        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                
                with gr.Tabs():
                    with gr.Tab("Text-to-Image-3D"):
                        txt_prompt = gr.Textbox(label="Prompt", placeholder="eg. A Plane 3D", lines=2)
                        btn_gen_img = gr.Button("1.Generate Image", variant="primary")
                    with gr.Tab("Image-to-3D"):
                        gr.Markdown("Upload an image directly if you have one.")
                
                image_prompt = gr.Image(label="Input Image", format="png", image_mode="RGBA", type="pil", height=350)

                with gr.Accordion(label="3D Settings", open=False):                     
                    resolution = gr.Radio(["512", "1024", "1536"], label="Generation Resolution", value="1024")
                    seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    
                decimation_target = gr.Slider(50000, 500000, label="Target Faces", value=150000, step=10000)
                texture_size = gr.Slider(512, 4096, label="Texture Size", value=1024, step=512)
                
                btn_gen_3d = gr.Button("2.Generate 3D", variant="primary", scale=2)

                with gr.Accordion(label="Advanced Sampler Settings", open=False):                
                    gr.Markdown("**Stage 1: Sparse Structure**")
                    ss_guidance_strength = gr.Slider(1.0, 10.0, value=7.5, label="Guidance")
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, value=0.7, label="Rescale")
                    ss_sampling_steps = gr.Slider(1, 50, value=12, label="Steps")
                    ss_rescale_t = gr.Slider(1.0, 6.0, value=5.0, label="Rescale T")
                    
                    gr.Markdown("**Stage 2: Shape**")
                    shape_guidance = gr.Slider(1.0, 10.0, value=7.5, label="Guidance")
                    shape_rescale = gr.Slider(0.0, 1.0, value=0.5, label="Rescale")
                    shape_steps = gr.Slider(1, 50, value=12, label="Steps")
                    shape_rescale_t = gr.Slider(1.0, 6.0, value=3.0, label="Rescale T")
                    
                    gr.Markdown("**Stage 3: Material**")
                    tex_guidance = gr.Slider(1.0, 10.0, value=1.0, label="Guidance")
                    tex_rescale = gr.Slider(0.0, 1.0, value=0.0, label="Rescale")
                    tex_steps = gr.Slider(1, 50, value=12, label="Steps")
                    tex_rescale_t = gr.Slider(1.0, 6.0, value=3.0, label="Rescale T")

            with gr.Column(scale=2):
                gr.Markdown("### 3D Output")
                
                rerun_output = Rerun(
                    label="Rerun 3D Viewer",
                    height=600
                )
                
                download_btn = gr.DownloadButton(label="3.Download GLB File", variant="primary")

                gr.Examples(
                    examples=[
                        ["example-images/A (1).webp"],
                        ["example-images/A (2).webp"],
                        ["example-images/A (3).webp"],
                        ["example-images/A (4).webp"],
                        ["example-images/A (5).webp"],
                        ["example-images/A (6).webp"],
                        ["example-images/A (7).webp"],
                        ["example-images/A (8).webp"],
                        ["example-images/A (9).webp"],
                        ["example-images/A (10).webp"],
                        ["example-images/A (11).webp"],
                        ["example-images/A (12).webp"],
                        ["example-images/A (13).webp"],
                        ["example-images/A (14).webp"],
                        ["example-images/A (15).webp"],
                        ["example-images/A (16).webp"],
                        ["example-images/A (17).webp"],
                        ["example-images/A (18).webp"],
                        ["example-images/A (19).webp"],
                        ["example-images/A (20).webp"],
                        ["example-images/A (21).webp"],
                        ["example-images/A (22).webp"],
                        ["example-images/A (23).webp"],
                        ["example-images/A (24).webp"],
                        ["example-images/A (25).webp"],
                        ["example-images/A (26).webp"],
                        ["example-images/A (27).webp"],
                        ["example-images/A (28).webp"],
                        ["example-images/A (29).webp"],
                        ["example-images/A (30).webp"],
                        ["example-images/A (31).webp"],
                        ["example-images/A (32).webp"],
                        ["example-images/A (33).webp"],
                        ["example-images/A (34).webp"],
                        ["example-images/A (35).webp"],
                        ["example-images/A (36).webp"],
                        ["example-images/A (37).webp"],
                        ["example-images/A (38).webp"],
                        ["example-images/A (39).webp"],
                        ["example-images/A (40).webp"],
                        ["example-images/A (41).webp"],
                        ["example-images/A (42).webp"],
                        ["example-images/A (43).webp"],
                        ["example-images/A (44).webp"],
                        ["example-images/A (45).webp"],
                        ["example-images/A (46).webp"],
                        ["example-images/A (47).webp"],
                        ["example-images/A (48).webp"],
                        ["example-images/A (49).webp"],
                        ["example-images/A (50).webp"],
                        ["example-images/A (51).webp"],
                        ["example-images/A (52).webp"],
                        ["example-images/A (53).webp"],
                        ["example-images/A (54).webp"],
                        ["example-images/A (55).webp"],
                        ["example-images/A (56).webp"],
                        ["example-images/A (57).webp"],
                        ["example-images/A (58).webp"],
                        ["example-images/A (59).webp"],
                        ["example-images/A (60).webp"],
                        ["example-images/A (61).webp"],
                        ["example-images/A (62).webp"],
                        ["example-images/A (63).webp"],
                        ["example-images/A (64).webp"],
                        ["example-images/A (65).webp"],
                        ["example-images/A (66).webp"],
                        ["example-images/A (67).webp"],
                        ["example-images/A (68).webp"],
                        ["example-images/A (69).webp"],
                        ["example-images/A (70).webp"],
                        ["example-images/A (71).webp"],
                    ],
                    inputs=[image_prompt],
                    label="Image Examples [image-to-3d]"
                )
                
                gr.Examples(
                    examples=[
                        ["A Cat 3D model"],
                        ["A realistic Cat 3D model"],
                        ["A cartoon Cat 3D model"],
                        ["A low poly Cat 3D"],
                        ["A cyberpunk Cat 3D"],
                        ["A robotic Cat 3D"],
                        ["A fluffy Cat 3D"],
                        ["A fantasy Cat 3D creature"],
                        ["A stylized Cat 3D"],
                        ["A Cat 3D sculpture"],
                
                        ["A Plane 3D model"],
                        ["A commercial Plane 3D"],
                        ["A fighter jet Plane 3D"],
                        ["A low poly Plane 3D"],
                        ["A vintage Plane 3D"],
                        ["A futuristic Plane 3D"],
                        ["A cargo Plane 3D"],
                        ["A private jet Plane 3D"],
                        ["A toy Plane 3D"],
                        ["A realistic Plane 3D"],
                
                        ["A Car 3D model"],
                        ["A sports Car 3D"],
                        ["A luxury Car 3D"],
                        ["A low poly Car 3D"],
                        ["A racing Car 3D"],
                        ["A cyberpunk Car 3D"],
                        ["A vintage Car 3D"],
                        ["A futuristic Car 3D"],
                        ["A SUV Car 3D"],
                        ["A electric Car 3D"],
                
                        ["A Shoe 3D model"],
                        ["A sneaker Shoe 3D"],
                        ["A running Shoe 3D"],
                        ["A leather Shoe 3D"],
                        ["A high heel Shoe 3D"],
                        ["A boot Shoe 3D"],
                        ["A low poly Shoe 3D"],
                        ["A futuristic Shoe 3D"],
                        ["A sports Shoe 3D"],
                        ["A casual Shoe 3D"],
                
                        ["A Chair 3D model"],
                        ["A Table 3D model"],
                        ["A Sofa 3D model"],
                        ["A Lamp 3D model"],
                        ["A Watch 3D model"],
                        ["A Backpack 3D model"],
                        ["A Drone 3D model"],
                        ["A Robot 3D model"],
                        ["A Smartphone 3D model"],
                        ["A Headphones 3D model"],
                
                        ["A House 3D model"],
                        ["A Skyscraper 3D model"],
                        ["A Bridge 3D model"],
                        ["A Castle 3D model"],
                        ["A Spaceship 3D model"],
                        ["A Rocket 3D model"],
                        ["A Satellite 3D model"],
                        ["A Tank 3D model"],
                        ["A Motorcycle 3D model"],
                        ["A Bicycle 3D model"]
                    ],
                    inputs=[txt_prompt],
                    label="3D Prompt Examples [text-to-3d]"
                )
                
        demo.load(start_session)
        demo.unload(end_session)

        btn_gen_img.click(
            generate_txt2img,
            inputs=[txt_prompt],
            outputs=[image_prompt]
        ).then(
            preprocess_image,
            inputs=[image_prompt],
            outputs=[image_prompt]
        )
        
        image_prompt.upload(
            preprocess_image,
            inputs=[image_prompt],
            outputs=[image_prompt],
        )

        btn_gen_3d.click(
            get_seed,
            inputs=[randomize_seed, seed],
            outputs=[seed],
        ).then(
            generate_3d,
            inputs=[
                image_prompt, seed, resolution, 
                decimation_target, texture_size,
                ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
                shape_guidance, shape_rescale, shape_steps, shape_rescale_t,
                tex_guidance, tex_rescale, tex_steps, tex_rescale_t,
            ],
            outputs=[rerun_output, download_btn],
        )

    demo.launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)
