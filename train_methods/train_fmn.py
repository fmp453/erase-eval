import os
import shutil
import warnings

from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline

from train_methods.utils_fmn import ti_component, attn_component
from utils import Arguments


warnings.filterwarnings("ignore")

def tokenize(s, tokenizer: CLIPTokenizer):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(s, add_special_tokens=False))
    if len(tokens) == 1:
        return tokens
    res = []
    for i in range(len(tokens)):
        if len(tokenizer.encode(tokens[i], add_special_tokens=False)) == 1:
            res.append(tokens[i])
        else:
            res = res + tokenize(tokens[i].replace("</w>", ""))
    return res

def make_initial_tokens(concept: str, tokenizer_version: str):
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(tokenizer_version, subfolder="tokenizer")

    RAND_TOKEN = "<rand-1>"

    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(concept, add_special_tokens=False))
    # ['car', 'av</w>', 'aggio</w>']

    l: list[str] = []
    for i in range(len(tokens)):
        if len(tokenizer.encode(tokens[i], add_special_tokens=False)) == 1:
            l.append(tokens[i])
        else:
            l = l + tokenize(tokens[i].replace("</w>", ""), tokenizer)

    for i in range(len(l)):
        l[i] = l[i].replace("</w>", "")
    # ['car', 'av', 'aggio']

    res = ""
    for i in range(len(l)):
        if len(l) - 1 != i:
            res += f"{l[i]}|"
        else:
            res += l[i]
    # car|av|aggio

    if len(l) < 4:
        N = 4 - len(l)
        res += "|"
        for i in range(N):
            if N - 1 != i:
                res += RAND_TOKEN + "|"
            else:
                res += RAND_TOKEN

    # car|av|aggio|<rand-1>
    return res

def make_placeholder_tokens(initializer_tokens: str):
    res = ""
    for i in range(n:=len(initializer_tokens.split("|"))):
        if n - 1 != i:
            res = res + f"<s{i+1}>|"
        else:
            res = res + f"<s{i+1}>"
    
    return res

def make_placeholder_token_at_data(placeholder_tokens: str):
    return "<s>|" + placeholder_tokens.replace("|", "")

def generation(args: Arguments):
    print("generate images for FMN")

    device = args.device.split(",")[0]
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_version).to(device=f"cuda:{device}")
    pipe.safety_checker = None
    prompt = f"a photo of {args.concepts}"
    os.makedirs(args.instance_data_dir, exist_ok=True)
    
    for i in range(2):
        images = pipe(prompt, num_images_per_prompt=5).images
        for j in range(5):
            images[j].save(f"{args.instance_data_dir}/{i:03}-{j}.png")

def main(args: Arguments):
    
    multi_concept = [[args.concepts, args.fmn_concept_type]]
    initializer_tokens = make_initial_tokens(args.concepts, args.sd_version)
    placeholder_tokens = make_placeholder_tokens(initializer_tokens)
    placeholder_token_at_data = make_placeholder_token_at_data(placeholder_tokens)

    device = f"cuda:{args.device.split(',')[0]}"

    generation(args)

    # Ti -> Attn の順で行う
    ti_component(
        instance_data_dir=args.instance_data_dir,
        pretrained_model_name_or_path=args.sd_version,
        output_dir=f"{args.save_dir}/{args.concepts}-ti",
        use_template=args.fmn_concept_type,
        placeholder_tokens=placeholder_tokens,
        placeholder_token_at_data=placeholder_token_at_data,
        initializer_tokens=initializer_tokens,
        class_prompt=None,
        seed=args.seed,
        resolution=args.image_size,
        color_jitter=False,
        train_batch_size=args.fmn_train_batch_size,
        max_train_steps_ti=args.fmn_max_train_steps_ti,
        save_steps=args.fmn_save_steps_ti,
        gradient_accumulation_steps=args.fmn_gradient_accumulation_steps,
        clip_ti_decay=args.clip_ti_decay,
        learning_rate_ti=args.fmn_lr_ti,
        scale_lr=args.fmn_scale_lr,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.fmn_lr_warmup_steps_ti,
        weight_decay_ti=args.fmn_weight_decay_ti,
        device=device,
        extra_args=args,
    )

    attn_component(
        instance_data_dir=args.instance_data_dir,
        seed=args.seed,
        # output_dir=args.save_dir,
        output_dir=f"{args.save_dir}/{args.concepts}-attn",
        pretrained_model_name_or_path=args.sd_version,
        multi_concept=multi_concept,
        scale_lr=args.fmn_scale_lr,
        learning_rate=args.fmn_lr_attn,
        gradient_accumulation_steps=args.fmn_gradient_accumulation_steps,
        train_batch_size=args.fmn_train_batch_size,
        only_optimize_ca=args.only_optimize_ca,
        resolution=args.image_size,
        center_crop=args.center_crop,
        use_pooler=args.use_pooler,
        dataloader_num_workers=args.fmn_dataloader_num_workers,
        max_train_steps=args.fmn_max_train_steps_attn,
        num_train_epochs=args.fmn_num_train_epochs,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.fmn_lr_warmup_steps_attn,
        lr_num_cycles=args.fmn_lr_num_cycles_attn,
        lr_power=args.fmn_lr_power_attn,
        no_real_image=False,
        max_grad_norm=args.max_grad_norm,
        device=device
    )

    shutil.rmtree(args.instance_data_dir)
