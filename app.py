import os
import gradio as gr
import json
import ast
import atexit
import shutil

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from gradio_image_prompter import ImagePrompter
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import numpy as np
from copy import deepcopy
import cv2

import sys
sys.path.append("libs")
sys.path.append("libs/LGM")
sys.path.append("libs/das")
sys.path.append("libs/sam2")

import torch.nn.functional as F
import torchvision
from torchvision import transforms
from einops import rearrange
import tempfile
import gc
from diffusers.utils import export_to_gif
import imageio
import sys
from sam2.sam2_image_predictor import SAM2ImagePredictor
from kiui.cam import orbit_camera
from src.utils.image_process import pred_bbox
from src.utils.load_utils import load_sv3d_pipeline, load_LGM, load_diffusion, gen_tracking_video, normalize_points, load_das
from src.utils.ui_utils import mask_image, image_preprocess, plot_point_cloud
from das.infer import load_media

import tyro
from tqdm import tqdm
from LGM.core.options import AllConfigs
from LGM.core.gs import GaussianRenderer
from LGM.mvdream.pipeline_mvdream import MVDreamPipeline

import h5py
os.environ["OMP_NUM_THREADS"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

segmentor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny", cache_dir="ckpt", device=device)

height, width = 480, 720
num_frames, sv3d_res = 20, 576
print(f"loading sv3d pipeline...")
sv3d_pipeline = load_sv3d_pipeline(device)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
sys.argv = ['pipeline_track_gen.py', 'big']
opt = tyro.cli(AllConfigs)
lgm_model = load_LGM(opt, device)

print(f'loading diffusion model...')
diffusion_model = load_diffusion(device=device, model_cfg_path='./src/configs/eval_base.yaml', diffusion_ckpt_path='./checkpoints/physctrl_base.safetensors')

temp_dir = tempfile.mkdtemp()
delete temp_dir after program exits
atexit.register(lambda: shutil.rmtree(temp_dir))
# temp_dir = './debug'
output_dir = temp_dir
print(f"using temp directory: {output_dir}")

print('loading das...')
das_model = load_das(0, output_dir)

import random
def set_all_seeds(seed):
    """Sets random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multiple GPUs

set_all_seeds(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def process_image(raw_input):
    image, points = raw_input['image'], raw_input['points']
    image = image.resize((width, height))
    image.save(f'{output_dir}/image.png')
    return image, {'image': image, 'points': points}

def segment(canvas, image, logits):
    if logits is not None:
        logits *=  32.0
    _, points = canvas['image'], canvas['points']
    image = np.array(image)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        segmentor.set_image(image)
        input_points = []
        input_boxes = []
        for p in points:
            [x1, y1, _, x2, y2, _] = p
            if x2==0 and y2==0:
                input_points.append([x1, y1])
            else:
                input_boxes.append([x1, y1, x2, y2])
        if len(input_points) == 0:
            input_points = None
            input_labels = None
        else:
            input_points = np.array(input_points)
            input_labels = np.ones(len(input_points))
        input_boxes = pred_bbox(Image.fromarray(image))
        if len(input_boxes) == 0:
            input_boxes = None
        else:
            input_boxes = np.array(input_boxes)
        masks, _, logits = segmentor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_boxes,
            multimask_output=False,
            return_logits=True,
            mask_input=logits,
        )
        mask = masks > 0
        masked_img = mask_image(image, mask[0], color=[252, 140, 90], alpha=0.9)
        masked_img = Image.fromarray(masked_img)
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = (
        mask.astype(np.uint8) * 255
    )
    out_image_bbox = Image.fromarray(out_image_bbox)
    y, x, res, sv3d_image = image_preprocess(out_image_bbox, target_res=sv3d_res, lower_contrast=False, rescale=True)
    np.save(f'{output_dir}/crop_info.npy', np.array([y, x, res]))
    print(f'crop_info: {y}, {x}, {res}')

    return mask[0], {'image': masked_img, 'points': points}, out_image_bbox, {'crop_y_start': y, 'crop_x_start': x, 'crop_res': res}, sv3d_image

def run_sv3d(image, seed=0):
    num_frames, sv3d_res = 20, 576
    elevations_deg = [0] * num_frames
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            if len(image.split()) == 4:  # RGBA
                input_image = Image.new("RGB", image.size, (255, 255, 255))  # pure white bg
                input_image.paste(image, mask=image.split()[3])  # 3rd is the alpha channel
            else:
                input_image = image
            
            video_frames = sv3d_pipeline(
                input_image.resize((sv3d_res, sv3d_res)),
                height=sv3d_res,
                width=sv3d_res,
                num_frames=num_frames,
                decode_chunk_size=8,  # smaller to save memory
                polars_rad=polars_rad,
                azimuths_rad=azimuths_rad,
                generator=torch.manual_seed(seed),
            ).frames[0]

    torch.cuda.empty_cache()
    gc.collect()

    # export_to_gif(video_frames, f"./debug/view_animation.gif", fps=7)
    for i, frame in enumerate(video_frames):
        # frame = frame.resize((res, res))
        frame.save(f"{output_dir}/{i:03d}.png")
    
    save_idx = [19, 4, 9, 14]
    for i in range(4):
        video_frames[save_idx[i]].save(f"{output_dir}/view_{i}.png")
    
    return [video_frames[i] for i in save_idx]

def run_LGM(image, seed=0):
    sv3d_frames = run_sv3d(image, seed)

    model = lgm_model
    rays_embeddings = model.prepare_default_rays(device)
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    images = []
    for i in range(4):
        # image = Image.open(f"{base_dir}/view_{i}.png")
        image = sv3d_frames[i]
        image = image.resize((256, 256))
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        images.append(image)
    mv_image = np.stack(images, axis=0)
    
    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, f'{output_dir}/point_cloud.ply')

        # render front view
        cam_poses = torch.from_numpy(orbit_camera(0, 0, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
        # cam_poses = torch.from_numpy(orbit_camera(45, 225, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
        np.save(f'{output_dir}/projection.npy', cam_view_proj[0].cpu().numpy())

        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
        image_save = (image[0, 0].permute(1, 2, 0).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(image_save).save(f'{output_dir}/front_view.png')

        images = []
        azimuth = np.arange(0, 360, 2, dtype=np.int32)
        elevation = 0
        
        for azi in tqdm(azimuth):
            
            cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
            
            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        out_video_dir = f'{output_dir}/gs_animation.mp4'    
        imageio.mimwrite(out_video_dir, images, fps=30)
        points, center, scale = normalize_points(output_dir)
        points_plot = plot_point_cloud(points, [])
        np.save(f'{output_dir}/center.npy', center)
        np.save(f'{output_dir}/scale.npy', scale)
        print('center: ', center, 'scale: ', scale)
    return points_plot, points

norm_fac = 5
mat_labels = {'elastic': 0, 'plasticine': 1, 'sand': 2, 'rigid': 3}

def run_diffusion(points, E_val, nu_val, x, y, z, u, v, w, force_coeff_val, floor_height=-1, fluid=False, seed=0, device='cuda'):
    drag_point = np.array([x, y, z])
    drag_dir = np.array([u, v, w])
    drag_dir /= np.linalg.norm(drag_dir)
    force_coeff = np.array(force_coeff_val)
    drag_force = drag_dir * force_coeff
    batch = {}
    
    batch['floor_height'] = torch.from_numpy(np.array([floor_height])).unsqueeze(-1).float()
    batch['points_src'] = (torch.from_numpy(points).float().unsqueeze(0) - norm_fac) / 2
    
    if not fluid:
        batch['drag_point'] = (torch.from_numpy(drag_point).float() - norm_fac) / 2
        batch['force'] = torch.from_numpy(np.array(drag_force)).float()
        batch['force'] = batch['force'] * torch.from_numpy(force_coeff) / torch.norm(batch['force'])
        batch['E'] = torch.from_numpy(np.array(E_val)).unsqueeze(-1).float()
        batch['nu'] = torch.from_numpy(np.array(nu_val)).unsqueeze(-1).float()
    else:
        batch['mask'] = torch.ones_like(batch['points_src'])
        batch['drag_point'] = torch.zeros(1, 3)
        batch['force'] = torch.zeros(1, 3)
        batch['E'] = torch.zeros(1, 1)
        batch['nu'] = torch.zeros(1, 1)
    
    for k in batch:
        batch[k] = batch[k].unsqueeze(0).to(device)
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = diffusion_model(batch['points_src'], batch['force'], batch['E'], batch['nu'], torch.ones_like(batch['points_src']).to(device)[..., :1],
            batch['drag_point'], batch['floor_height'], gravity=None, y=None, coeff=batch['E'], device=device, batch_size=1,
            generator=torch.Generator().manual_seed(seed), n_frames=24, num_inference_steps=25)
        output = output.cpu().numpy()
        for j in range(output.shape[0]):
            # save_pointcloud_video(((output[j:j+1] * 2) + norm_fac).squeeze(), [], f'{output_dir}/gen_animation.gif', grid_lim=10)
            np.save(f'{output_dir}/gen_data.npy', output[j:j+1].squeeze())
    gen_tracking_video(output_dir)
    return os.path.join(output_dir, 'tracks_gen/tracking/tracks_tracking.mp4')

def run_diffusion_new(points, E_val, nu_val, x, y, z, u, v, w, force_coeff_val, material='elastic', drag_mode='point', drag_axis='z', seed=0, device='cuda'):
    drag_point = np.array([x, y, z])
    drag_dir = np.array([u, v, w])
    # User input
    has_gravity = (material != 'elastic')
    force_coeff = np.array(force_coeff_val)
    max_num_forces = 1
    if drag_mode is not None and not has_gravity:
        if drag_mode == "point":
            drag_point = np.array(drag_point)
        elif drag_mode == "max":
            drag_point_idx = np.argmax(points[:, drag_axis]) if drag_mode == "max" \
                else np.argmin(points[:, drag_axis])
            drag_point = points[drag_point_idx]
        else:
            raise ValueError(f"Invalid drag mode: {drag_mode}")
        drag_offset = np.abs(points - drag_point)
        drag_mask = (drag_offset < 0.4).all(axis=-1)
        drag_dir = np.array(drag_dir, dtype=np.float32)
        drag_dir /= np.linalg.norm(drag_dir)
        drag_force = drag_dir * force_coeff
    else:
        drag_mask = np.ones(N, dtype=bool)
        drag_point = np.zeros(4)
        drag_dir = np.zeros(3)
        drag_force = np.zeros(3) 
    
    if material == "elastic":
        log_E, nu = np.array(E_val), np.array(nu_val)
    else: 
        log_E, nu = np.array(6), np.array(0.4) # Default values for non-elastic materials

    print(f'[Diffusion Simulation] Number of drag points: {drag_mask.sum()}/{2048}')
    print(f'[Diffusion Simulation] Drag point: {drag_point}')
    print(f'[Diffusion Simulation] log_E: {log_E}, ν: {nu}')
    print(f'[Diffusion Simulation] Drag force: {drag_force}')
    print(f'[Diffusion Simulation] Material type: {material})')
    print(f'[Diffusion Simulation] Has gravity: {has_gravity}')

    force_order = torch.arange(max_num_forces) 
    mask = torch.from_numpy(drag_mask).bool()
    mask = mask.unsqueeze(0) if mask.ndim == 1 else mask  
     
    batch = {} 
    batch['gravity'] = torch.from_numpy(np.array(has_gravity)).long().unsqueeze(0)
    batch['drag_point'] = torch.from_numpy(drag_point - norm_fac).float() / 2
    batch['drag_point'] = batch['drag_point'].unsqueeze(0) # (1, 4)
    batch['points_src'] = (torch.from_numpy(points).float().unsqueeze(0) - norm_fac) / 2

    if has_gravity:
        floor_normal = np.load(f'{output_dir}/floor_normal.npy')
        floor_height = np.load(f'{output_dir}/floor_height.npy') * scale / 2.
        batch['floor_height'] = torch.from_numpy(np.array(floor_height)).float().unsqueeze(0)

        # Create rotation matrix to align floor normal with [0, 1, 0] (upward direction)
        target_normal = np.array([0, 1, 0])
        
        # Use Rodrigues' rotation formula to find rotation matrix
        # Rotate from floor_normal to target_normal
        v = np.cross(floor_normal, target_normal)
        s = np.linalg.norm(v)
        c = np.dot(floor_normal, target_normal)
        
        if s < 1e-6:  # If vectors are parallel
            if c > 0:  # Same direction
                R_floor = np.eye(3)
            else:  # Opposite direction
                R_floor = -np.eye(3)
        else:
            v = v / s
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_floor = np.eye(3) + s * K + (1 - c) * (K @ K)

        R_floor_tensor = torch.from_numpy(R_floor).float().to(device)
        for i in range(batch['points_src'].shape[0]):
            batch['points_src'][i] = (R_floor_tensor @ batch['points_src'][i].T).T
    else:
        batch['floor_height'] = torch.ones(1).float() * -2.4

    print(f'[Diffusion Simulation] Floor height: {batch["floor_height"]}')

    if mask.shape[1] == 0:
        mask = torch.zeros(0, N).bool()
        batch['force'] = torch.zeros(0, 3)
        batch['drag_point'] = torch.zeros(0, 4) 
    else:
        batch['force'] = torch.from_numpy(drag_force).float().unsqueeze(0)
        batch['force'] = batch['force'] * torch.from_numpy(force_coeff) / torch.norm(batch['force'])
     
    batch['mat_type'] = torch.from_numpy(np.array(mat_labels[material])).long()
    if np.array(batch['mat_type']).item() == 3: # Rigid dataset
        batch['is_mpm'] = torch.tensor(0).bool()
    else:
        batch['is_mpm'] = torch.tensor(1).bool()
    
    if has_gravity: # Currently we only have either drag force or gravity  
        batch['force'] = torch.tensor([[0, -1.0, 0]]).to(device)   
    
    all_forces = torch.zeros(max_num_forces, 3)
    all_forces[:batch['force'].shape[0]] = batch['force']
    all_forces = all_forces[force_order]
    batch['force'] = all_forces

    all_drag_points = torch.zeros(max_num_forces, 4)  
    all_drag_points[:batch['drag_point'].shape[0], :batch['drag_point'].shape[1]] = batch['drag_point'] # The last dim of drag_point is not used now
    all_drag_points = all_drag_points[force_order]
    batch['drag_point'] = all_drag_points

    if batch['gravity'][0] == 1: # add gravity to force
        batch['force'] = torch.tensor([[0, -1.0, 0]]).float().to(device) 

    all_mask = torch.zeros(max_num_forces, 2048).bool()
    all_mask[:mask.shape[0]] = mask
    all_mask = all_mask[force_order]

    batch['mask'] = all_mask[..., None] # (n_forces, N, 1) for compatibility
    batch['E'] = torch.from_numpy(log_E).unsqueeze(-1).float() if log_E > 0 else torch.zeros(1).float()
    batch['nu'] = torch.from_numpy(nu).unsqueeze(-1).float()

    for k in batch:
        batch[k] = batch[k].unsqueeze(0).to(device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = diffusion_model(batch['points_src'], batch['force'], batch['E'], batch['nu'], batch['mask'][..., :1],
            batch['drag_point'], batch['floor_height'], batch['gravity'], coeff=batch['E'], generator=torch.Generator().manual_seed(seed), 
            device=device, batch_size=1, y=batch['mat_type'], n_frames=24, num_inference_steps=25)
        output = output.cpu().numpy()  
        for j in range(output.shape[0]):
            if batch['gravity'][0] == 1:
                for k in range(output.shape[1]):
                    output[j, k] = (np.linalg.inv(R_floor) @ output[j, k].T).T 
            np.save(f'{output_dir}/gen_data.npy', output[j:j+1].squeeze())
    gen_tracking_video(output_dir)
    return os.path.join(output_dir, 'tracks_gen/tracking/tracks_tracking.mp4')

def run_das(prompt, tracking_path, checkpoint_path='./checkpoints/cogshader5B'):
    print(prompt, tracking_path)
    input_path = os.path.join(output_dir, 'image.png')
    video_tensor, fps, is_video = load_media(input_path)
    tracking_tensor, _, _ = load_media(tracking_path)
    das_model.apply_tracking(
        video_tensor=video_tensor,
        fps=24,
        tracking_tensor=tracking_tensor,
        img_cond_tensor=None,
        prompt=prompt,
        checkpoint_path=checkpoint_path
    )
    return os.path.join(output_dir, 'result.mp4')

def add_arrow(points, x, y, z, u, v, w, force_coeff):
    direction = np.array([u, v, w])
    direction /= np.linalg.norm(direction)
    arrow = {'origin': [x, y, z], 'dir': direction * force_coeff}
    arrows = [arrow]
    points_plot = plot_point_cloud(points, arrows)
    return points_plot

material_slider_config = {
    "Elastic": [
        {"label": "E", "minimum": 4, "maximum": 7, "step": 0.5, "value": 5.5},
        {"label": "nu", "minimum": 0.2, "maximum": 0.4, "step": 0.05, "value": 0.3},
    ],
    "Plasticine": [
        {"label": "E", "minimum": 4, "maximum": 7, "step": 0.5, "value": 5.5},
        {"label": "nu", "minimum": 0.2, "maximum": 0.4, "step": 0.05, "value": 0.3},
    ],
    "Plastic": [
        {"label": "E", "minimum": 4, "maximum": 7, "step": 0.5, "value": 5.5},
        {"label": "nu", "minimum": 0.2, "maximum": 0.4, "step": 0.05, "value": 0.3},
    ],
    "Rigid": []  # No sliders
}

def update_sliders(material):
    sliders = material_slider_config[material]
    # Prepare updates for both sliders
    if len(sliders) == 2:
        return (
            gr.update(visible=True, interactive=True, **sliders[0]),
            gr.update(visible=True, interactive=True, **sliders[1])
        )
    elif len(sliders) == 1:
        return (
            gr.update(visible=True, interactive=True, **sliders[0]),
            gr.update(visible=False, interactive=False)
        )
    else:
        return (
            gr.update(visible=False, interactive=False),
            gr.update(visible=False, interactive=False)
        )
update_sliders('Elastic')

with gr.Blocks() as demo:
    mask = gr.State(value=None) # store mask
    original_image = gr.State(value=None) # store original input image
    mask_logits = gr.State(value=None) # store mask logits
    masked_image = gr.State(value=None) # store masked image
    crop_info = gr.State(value=None) # store crop info
    sv3d_input = gr.State(value=None) # store sv3d input
    sv3d_frames = gr.State(value=None) # store sv3d frames
    points = gr.State(value=None) # store points

    with gr.Column():
        with gr.Row():
            with gr.Column():
                step1_dec = """
                    <font size="4"><b>Step 1: Upload Input Image and Segment Subject</b></font>
                    """
                step1 = gr.Markdown(step1_dec)
                raw_input = ImagePrompter(type="pil", label="Input Image", show_label=True, interactive=True)
                process_button = gr.Button("Process")
                
            with gr.Column():
                # Step 2: Get Subject Mask and Point Clouds
                step2_dec = """
                    <font size="4"><b>Step 2.1: Get Subject Mask</b></font>
                    """
                step2 = gr.Markdown(step2_dec)
                canvas = ImagePrompter(type="pil", label="Input Image", show_label=True, interactive=True) # for mask painting

                step2_notes = """
                    - Click to add points to select the subject.
                    - Press `Segment Subject` to get the mask. <mark>Can be refined iteratively by updating points<mark>.
                """
                notes = gr.Markdown(step2_notes)
                segment_button = gr.Button("Segment Subject") 

            # with gr.Column():
            #     output_video = gr.Video(label="Rendered Video", format="mp4", width="auto", autoplay=True, interactive=False)
            with gr.Column(scale=1):
                step22_dec = """
                    <font size="4"><b>Step 2.2: Get 3D Points</b></font>
                    """
                step22 = gr.Markdown(step22_dec)
                points_plot = gr.Plot(label="Point Cloud")
                sv3d_button = gr.Button("Get 3D Points")
            
            with gr.Column():
                step3_dec = """
                    <font size="4"><b>Step 3: Add Force</b></font>
                    """
                step3 = gr.Markdown(step3_dec) 
                with gr.Row():
                    gr.Markdown('Add Drag Point')
                with gr.Row():
                    x = gr.Number(label="X", min_width=50)
                    y = gr.Number(label="Y", min_width=50)
                    z = gr.Number(label="Z", min_width=50)
                with gr.Row():
                    gr.Markdown('Add Drag Direction')
                with gr.Row():
                    u = gr.Number(label="U", min_width=50)
                    v = gr.Number(label="V", min_width=50)
                    w = gr.Number(label="W", min_width=50)
                step3_notes = """
                    <b>Direction will be normalized to unit length.</b>
                """
                notes = gr.Markdown(step3_notes)
                with gr.Row():
                    force_coeff = gr.Slider(label="Force Magnitude", minimum=0.02, maximum=0.2, step=0.02, value=0.045)
                add_arrow_button = gr.Button("Add Force")
                
        with gr.Row():

            with gr.Column():
                step4_dec = """
                    <font size="4"><b>Step 4: Select Material and Generate Trajectory</b></font>
                    """
                step4 = gr.Markdown(step4_dec)
                tracking_video = gr.Video(label="Tracking Video", format="mp4", width="auto", autoplay=True, interactive=False)
                with gr.Row():
                #     material_radio = gr.Radio(
                #         choices=list(material_slider_config.keys()),
                #         label="Choose Material",
                #         value="Rigid"
                #     )      
                    # slider1 = gr.Slider(visible=True)
                    # slider2 = gr.Slider(visible=True)
                    slider1 = gr.Slider(label="E", visible=True, interactive=True, minimum=4, maximum=7, step=0.5, value=5.5)
                    slider2 = gr.Slider(visible=False, minimum=0.2, maximum=0.4, step=0.05, value=0.3)
                run_diffusion_button = gr.Button("Generate Trajectory")

            with gr.Column():
                step5_dec = """
                    <font size="4"><b>Step 5: Generate Final Video</b></font>
                    """
                step5 = gr.Markdown(step5_dec)
                final_video = gr.Video(label="Final Video", format="mp4", width="auto", autoplay=True, interactive=False)
                text = gr.Textbox(label="Prompt")
                gen_video_button = gr.Button("Generate Final Video")
                            
    
    # material_radio.change(
    #     fn=update_sliders,
    #     inputs=material_radio,
    #     outputs=[slider1, slider2]
    # )
    process_button.click(
        fn = process_image,
        inputs = [raw_input],
        outputs = [original_image, canvas]
    )
    segment_button.click(
        fn = segment,
        inputs = [canvas, original_image, mask_logits],
        outputs = [mask, canvas, masked_image, crop_info, sv3d_input]
    )
    sv3d_button.click(
        fn = run_LGM,
        inputs = [sv3d_input],
        outputs = [points_plot, points]
    )
    add_arrow_button.click(
        fn=add_arrow,
        inputs=[points, x, y, z, u, v, w, force_coeff],
        outputs=points_plot
    )
    run_diffusion_button.click(
        fn=run_diffusion_new,
        inputs=[points, slider1, slider2, x, y, z, u, v, w, force_coeff],
        outputs=tracking_video
    )
    gen_video_button.click(
        fn=run_das,
        inputs=[text, tracking_video],
        outputs=final_video
    )
demo.queue().launch(share=True)
