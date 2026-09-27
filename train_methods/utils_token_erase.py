import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms
from pathlib import Path

class MultiReferenceSelfAttention():
    def __init__(
        self,
        start_step=0,
        end_step=50,
        step_idx=None,
        layer_idx=None,
        ref_masks=None,
        mask_weights=[1.0,1.0,1.0],
        style_fidelity=1,
        viz_cfg=None
    ):
        """
        Args:
            start_step   : the step to start transforming self-attention to multi-reference self-attention
            end_step     : the step to end transforming self-attention to multi-reference self-attention
            step_idx     : list of the steps to transform self-attention to multi-reference self-attention
            layer_idx    : list of the layers to transform self-attention to multi-reference self-attention
            ref_masks    : masks of the input reference images
            mask_weights : mask weights for each reference masks
            viz_cfg      : config for visualization
        """
        self.cur_step =  0
        self.num_att_layers = -1
        self.cur_att_layer =  0
        self.start_step = start_step
        self.end_step = end_step
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.layer_idx = layer_idx
        self.ref_masks = ref_masks
        self.mask_weights = mask_weights
        self.style_fidelity = style_fidelity
        self.viz_cfg = viz_cfg
        self.enabled = False
        self.use_reference = False

    def __call__(self, q, k, v, attn, is_cross, num_heads, **kwargs):
        if not self.enabled or not self.use_reference or is_cross:
            out = self.sa_forward(v, attn, num_heads, **kwargs)
        else:
            out = self.mrsa_forward(q, k, v, attn, is_cross, num_heads, **kwargs)

        if self.enabled:
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                
        return out
    
    def get_ref_mask(self, ref_mask, mask_weight, H, W):
        ref_mask = ref_mask.float() * mask_weight
        ref_mask = F.interpolate(ref_mask, (H, W))
        ref_mask = ref_mask.flatten()
        return ref_mask
    
    def attn_batch(self, q, k, v, sim, attn, num_heads, **kwargs):
        # 确保 dtype 对齐，避免 Half/Float 混用
        if attn is not None and v is not None and attn.dtype != v.dtype:
            v = v.to(attn.dtype)
        if q is not None and k is not None and q.dtype != k.dtype:
            k = k.to(q.dtype)
            if v is not None and v.dtype != q.dtype:
                v = v.to(q.dtype)

        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads) 
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        
        if kwargs.get("attn_batch_type") == 'mrsa':
            sim_own, sim_refs = sim[..., :H*W], sim[..., H*W:]
            sim_or = [sim_own]
            for i, (ref_mask, mask_weight) in enumerate(zip(self.ref_masks, self.mask_weights)):
                ref_mask = self.get_ref_mask(ref_mask, mask_weight, H, W)
                sim_ref = sim_refs[..., H*W*i: H*W*(i+1)]
                sim_ref = sim_ref + ref_mask.masked_fill(ref_mask == 0, torch.finfo(sim.dtype).min)
                sim_or.append(sim_ref)
            sim = torch.cat(sim_or, dim=-1)
        attn = sim.softmax(-1)
        attn = attn.to(v.dtype)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out  

    def sa_forward(self, v, attn, num_heads, **kwargs):
        """
        Original self-attention forward function
        """
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out
    
    def mrsa_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn, is_cross, num_heads, **kwargs):
        """
        Mutli-reference self-attention(MRSA) forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return self.sa_forward(v, attn, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2

        if B <= 1:
            return self.sa_forward(v, attn, num_heads, **kwargs)
        
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        qu_o = qu[:num_heads]
        qc_o = qc[:num_heads]
        
        ku_o, ku_r = ku[:num_heads], ku[num_heads:]
        kc_o, kc_r = kc[:num_heads], kc[num_heads:]
        
        vu_o, vu_r = vu[:num_heads], vu[num_heads:]
        vc_o, vc_r = vc[:num_heads], vc[num_heads:]
        ku_cat, vu_cat = torch.cat([ku_o, *ku_r.chunk(B-1)], 1), torch.cat([vu_o, *vu_r.chunk(B-1)], 1)
        kc_cat, vc_cat = torch.cat([kc_o, *kc_r.chunk(B-1)], 1), torch.cat([vc_o, *vc_r.chunk(B-1)], 1)

        out_u_target = self.attn_batch(qu_o, ku_cat, vu_cat, None, None, num_heads, attn_batch_type='mrsa', **kwargs)
        out_c_target = self.attn_batch(qc_o, kc_cat, vc_cat, None, None, num_heads, attn_batch_type='mrsa', **kwargs)
        
        # The larger the style_fidelity, the more like the reference concepts, range of values: [0,1]
        if self.style_fidelity > 0:
            out_u_target = (1 - self.style_fidelity) * out_u_target + self.style_fidelity * self.attn_batch(qu_o, ku_o, vu_o, None, None, num_heads, **kwargs)

        out = self.sa_forward(v, attn, num_heads, **kwargs)
        out_u, out_c = out.chunk(2)
        out_u_ref, out_c_ref = out_u[1:], out_c[1:]
        return torch.cat([out_u_target, out_u_ref, out_c_target, out_c_ref], dim=0)


def load_image(image_path, width=512, height=512):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    if width and height:
        image = image.resize((width, height))
    return image

def load_mask(mask_path, width=512, height=512):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    mask = Image.open(mask_path).convert("L")  # 转为灰度图
    if width and height:
        mask = mask.resize((width, height))
    return mask

def load_reference_config(config_path_or_dir):
    # 这个函数现在主要用于向后兼容，实际不应该被调用
    if not config_path_or_dir:
        print("[DEBUG] Config path is None or empty")
        return None
        
    if not os.path.exists(config_path_or_dir):
        print(f"[DEBUG] Config path does not exist: {config_path_or_dir}")
        return None
    
    if os.path.isfile(config_path_or_dir):
        print(f"[DEBUG] Config path is a file, trying to load as YAML: {config_path_or_dir}")
        try:
            with open(config_path_or_dir, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[DEBUG] Successfully loaded YAML config")
            return config
        except Exception as e:
            print(f"[DEBUG] Error loading config file {config_path_or_dir}: {e}")
            return None
    
    # 不再支持目录扫描
    print(f"[WARNING] Directory-based reference image configuration is no longer supported")
    print(f"[WARNING] Please configure reference images in your YAML file under 'reference_images' section")
    return None

def create_mrsa_from_references(ref_data, mrsa_config):
    """基于参考图像数据创建MRSA对象"""
    if not ref_data:
        print("[ERROR] create_mrsa_from_references: ref_data is None")
        return None
    
    try:
        mrsa = MultiReferenceSelfAttention(
            start_step=getattr(mrsa_config, 'mrsa_start_step', 0),
            end_step=getattr(mrsa_config, 'mrsa_end_step', 50),
            layer_idx=getattr(mrsa_config, 'mrsa_layer_idx', [0, 1, 2]),
            ref_masks=ref_data["ref_masks"],
            mask_weights=ref_data["mask_weights"],
            style_fidelity=getattr(mrsa_config, 'style_fidelity', 0.1)
        )
        
        print(f"[INFO] Created MRSA with parameters:")
        print(f"  - start_step: {mrsa.start_step}")
        print(f"  - end_step: {mrsa.end_step}")
        print(f"  - layer_idx: {mrsa.layer_idx}")
        print(f"  - style_fidelity: {mrsa.style_fidelity}")
        print(f"  - num_ref_masks: {len(mrsa.ref_masks)}")
        
        return mrsa
        
    except Exception as e:
        print(f"[ERROR] Failed to create MRSA: {e}")
        import traceback
        traceback.print_exc()
        return None

def auto_pair_images_and_masks(image_dir, mask_dir, image_pattern="*.jpg", mask_pattern="*_mask.jpg", mask_suffix="_mask"):
    """自动配对图片和mask文件"""
    
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir) if mask_dir else None
    
    if not image_dir.exists():
        print(f"[ERROR] Image directory not found: {image_dir}")
        return [], []
    
    # 如果没有mask目录，返回所有图片，mask设为None
    if not mask_dir or not mask_dir.exists():
        # print(f"[INFO] No mask directory provided or not found, will use full-image masks")
        image_files = list(image_dir.glob(image_pattern))
        image_files.sort()
        paired_images = [str(img_path) for img_path in image_files]
        paired_masks = [None] * len(paired_images)  # 所有mask都设为None
        return paired_images, paired_masks
    
    # 获取所有图片文件
    image_files = list(image_dir.glob(image_pattern))
    image_files.sort()  # 确保顺序一致
    
    paired_images = []
    paired_masks = []
    
    for img_path in image_files:
        # 提取基础文件名（不含扩展名）
        base_name = img_path.stem
        
        # 构造对应的mask文件名
        mask_name = f"{base_name}{mask_suffix}{img_path.suffix}"
        mask_path = mask_dir / mask_name
        
        paired_images.append(str(img_path))
        
        if mask_path.exists():
            paired_masks.append(str(mask_path))
            # print(f"[INFO] Paired: {img_path.name} <-> {mask_path.name}")
        else:
            # print(f"[INFO] No mask found for {img_path.name}, will use full-image mask")
            paired_masks.append(None)  # None表示需要创建全图mask
    
    return paired_images, paired_masks

def create_full_image_mask(latent_shape, device, dtype=torch.float32):
    """创建全图mask"""
    _, _, h, w = latent_shape
    full_mask = torch.ones((1, 1, h, w), device=device, dtype=dtype)
    return full_mask

def process_reference_images(ref_config, vae, device, weight_dtype):
    """自动配对处理参考图像 - 修改版：支持无mask时创建全图mask"""
    
    # 检查是否使用新的配置方式
    if 'image_dir' in ref_config:
        # 新方式：自动配对
        image_dir = ref_config['image_dir']
        mask_dir = ref_config.get('mask_dir')  # 可能为None
        image_pattern = ref_config.get('image_pattern', '*.jpg')
        mask_pattern = ref_config.get('mask_pattern', '*_mask.jpg')
        mask_suffix = ref_config.get('mask_suffix', '_mask')
        
        # 支持多种图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        all_image_paths = []
        
        for ext in image_extensions:
            image_paths_for_ext, _ = auto_pair_images_and_masks(
                image_dir, mask_dir, ext, mask_pattern, mask_suffix
            )
            all_image_paths.extend(image_paths_for_ext)
        
        # 如果用指定的pattern没找到图片，尝试所有扩展名
        if not all_image_paths:
            for ext in image_extensions:
                image_paths_for_ext, mask_paths_for_ext = auto_pair_images_and_masks(
                    image_dir, mask_dir, ext, mask_pattern, mask_suffix
                )
                if image_paths_for_ext:
                    image_paths = image_paths_for_ext
                    mask_paths = mask_paths_for_ext
                    break
            else:
                print(f"[ERROR] No images found in {image_dir}")
                return None
        else:
            # 重新配对获取mask路径
            image_paths, mask_paths = auto_pair_images_and_masks(
                image_dir, mask_dir, image_pattern, mask_pattern, mask_suffix
            )
        
        # 处理prompts
        base_prompts = ref_config.get('prompts', [])
        if base_prompts:
            # 为每张图片分配prompts（循环使用）
            prompts = []
            for i in range(len(image_paths)):
                prompt_idx = i % len(base_prompts)
                prompts.append(base_prompts[prompt_idx])
        else:
            prompts = []
        
        # 处理mask权重
        default_weight = ref_config.get('default_mask_weight', 1.0)
        custom_weights = ref_config.get('mask_weights', {})
        mask_weights = []
        
        for img_path in image_paths:
            base_name = Path(img_path).stem
            # 检查是否有自定义权重
            weight = custom_weights.get(base_name, default_weight)
            mask_weights.append(weight)
    
    print(f"[INFO] Found {len(image_paths)} images to process")
    
    if not image_paths:
        print("[WARNING] No image paths found")
        return None
    
    ref_images = []
    ref_latents_z_0 = []
    ref_masks = []
    processed_mask_weights = []
    processed_prompts = []
    
    # 统计mask类型
    custom_mask_count = 0
    auto_mask_count = 0
    
    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"[WARNING] Reference image not found: {img_path}")
            continue
            
        
        # 加载图像
        image = load_image(img_path, 512, 512)
        ref_images.append(image)
        
        # 编码为latents
        with torch.no_grad():
            image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device, dtype=weight_dtype)
            image_tensor = image_tensor * 2.0 - 1.0
            latents = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor
            ref_latents_z_0.append(latents)
        
        # 处理mask - 关键修改：支持自动创建全图mask
        mask_tensor = None
        if i < len(mask_paths) and mask_paths[i] is not None:
            mask_path = mask_paths[i]
            if os.path.exists(mask_path):
                # try:
                mask = load_mask(mask_path, 512, 512)
                mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(device, dtype=weight_dtype)
                
                # 将mask调整到与latent相同的空间尺寸
                _, _, h, w = latents.shape
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor, size=(h, w), mode='bilinear', align_corners=False
                )
                
                # 确保mask值在[0,1]范围内
                mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)
                
                custom_mask_count += 1
                    # print(f"[INFO] Loaded custom mask: {Path(mask_path).name}")
                # except Exception as e:
                #     print(f"[WARNING] Failed to load mask {mask_path}: {e}")
                #     mask_tensor = None
            else:
                print(f"[WARNING] Mask file not found: {mask_path}")
                mask_tensor = None
        
        # 如果没有mask或mask加载失败，创建全图mask
        if mask_tensor is None:
            mask_tensor = create_full_image_mask(latents.shape, device, weight_dtype)
            auto_mask_count += 1
            # print(f"[INFO] Created full-image mask for: {Path(img_path).name}")
        
        ref_masks.append(mask_tensor)
        
        # 处理权重
        if i < len(mask_weights):
            processed_mask_weights.append(mask_weights[i])
        else:
            processed_mask_weights.append(1.0)
        
        # 处理prompts
        if i < len(prompts):
            processed_prompts.append(prompts[i])
        else:
            # 如果prompts不够，使用默认prompt
            base_name = Path(img_path).stem
            processed_prompts.append(f"a photo of {base_name}")
            
        # except Exception as e:
        #     print(f"[ERROR] Failed to process reference image {img_path}: {e}")
        #     continue
    
    if not ref_images:
        print("[ERROR] No valid reference images processed")
        return None
    
    # 打印处理结果统计
    print(f"[INFO] Successfully processed {len(ref_images)} reference images")
    print(f"[INFO] Mask statistics:")
    print(f"  - Custom masks loaded: {custom_mask_count}")
    print(f"  - Full-image masks created: {auto_mask_count}")
    
    if mask_dir:
        print(f"[INFO] Mask directory: {mask_dir}")
    else:
        print(f"[INFO] No mask directory provided - all images use full-image masks")
    
    return {
        "ref_images": ref_images,
        "ref_latents_z_0": ref_latents_z_0,
        "ref_masks": ref_masks,
        "mask_weights": processed_mask_weights,
        "ref_prompts": processed_prompts
    }