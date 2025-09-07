import argparse
import importlib

from typing import Optional, Literal
from pydantic import BaseModel, Field

def execute_function(method, mode):
    module_name = f"{mode}_methods.{mode}_{method}"

    try:
        train_module = importlib.import_module(module_name)
        train_function = getattr(train_module, "main")
    except ModuleNotFoundError:
        print(f"Module {module_name} not found.")
        exit(1)
    except AttributeError:
        print(f"Function 'main' not found in module {module_name}.")
        exit(1)
    return train_function

def get_args():
    args = Arguments.parse_args()
    return args

class Arguments(BaseModel):
    
    mode: Literal["train", "infer"] = Field("train", description="train (erase) or infer")
    method: Literal["esd", "ac", "eap", "adv", "locogen", "uce", "mace", "receler", "fmn", "salun", "spm", "sdd", "diffquickfix", "doco", "gloce", "age", "original"] = Field("esd")
    sd_version: Optional[str] = Field("compvis/stable-diffusion-v1-4")
    device: Optional[str] = Field("0", description="gpu id. when using two gpus, separated by comma")
    seed: Optional[int] = Field(0)

    # training part
    # general configs
    concepts: Optional[str] = Field("English springer", description="separated by comma")
    save_dir: Optional[str] = Field("models", description="path to dir for erased models")

    anchor_concept: Optional[str] = Field("dog")
    seperator: Optional[str] = Field(None, desciption='separator if you want to train bunch of erased_words separately')

    image_size: Optional[int] = Field(512, desciption='image size used to train')
    ddim_steps: Optional[int] = Field(50, desciption='ddim steps of inference used to train')
    ddpm_steps: Optional[int] = Field(1000)
    max_grad_norm: Optional[float] = Field(1.0)
    lr_scheduler: Literal["constant", "linear","cosine", "cosine_warmup", "cosine_warmup_restart", "polynomial", "polynomial_warmup", "polynomial_warmup_restart"] = Field("constant", description="learning rate scheduler")
    
    negative_guidance: Optional[float] = Field(1, desciption='guidance of negative training used to train')
    start_guidance: Optional[float] = Field(3, desciption='guidance of start image used to train')

    # for EAP and AGE
    gumbel_lr: Optional[float] = Field(1e-3, desciption='learning rate for prompt')
    gumbel_temp: Optional[float] = Field(2, desciption='temperature for gumbel softmax')
    gumbel_hard: Literal[0, 1] = Field(0, desciption='hard for gumbel softmax, 0: soft, 1: hard')
    gumbel_num_centers: Optional[int] = Field(100, desciption='number of centers for kmeans, if <= 0 then do not apply kmeans')
    gumbel_update: Optional[int] = Field(100, desciption='update frequency for preserved set, if <= 0 then do not update')
    gumbel_time_step: Optional[int] = Field(0, desciption='time step for the starting point to estimate epsilon')
    gumbel_multi_steps: Optional[int] = Field(2, desciption='multi steps for calculating the output')
    gumbel_k_closest: Optional[int] = Field(1000, desciption='number of closest tokens to consider')
    ignore_special_tokens: Literal[True, False] = Field(True, desciption='ignore special tokens in the embedding matrix')
    vocab: Optional[str] = Field("EN3K", desciption='vocab')
    pgd_num_steps: Optional[int] = Field(2, desciption='number of step to optimize adversarial concepts')

    # configs for ESD (Erased Stable Diffusion)
    esd_method: Literal["full", "selfattn", "xattn", "noxattn", "notime"] = Field("xattn", description="which parameters are updated")
    esd_iter: Optional[int] = Field(1000)
    esd_lr: Optional[float] = Field(1e-5)
    esd_eta: Optional[float] = Field(0.0)
    esd_lr_warmup_steps: Optional[int] = Field(500)
    

    # configs for AC (Ablating Concepts)
    ac_method: Literal["full", "xattn"] = Field("xattn", description="which parameters are updated")
    ac_lr: Optional[float] = Field(2e-6)
    
    ac_img_dir: Optional[str] = Field("images")
    ac_prompt_path: Optional[str] = Field("dog.csv")
    ac_concept_type: Literal["object", "style", "mem"] = Field("object")
    ac_batch_size: Optional[int] = Field(8)
    

    # configs for EAP (Erasing-Adversarial-Preservation)
    eap_method: Literal["full", "selfattn", "xattn", "noxattn", "notime", "xattn_matching", "xlayer", "selflayer"] = Field("xattn", desciption='method of training')
    eap_iterations: Optional[int] = Field(1000, desciption='iterations used to train')
    eap_lr: Optional[float] = Field(1e-5, desciption='learning rate used to train')

    # configs for AdvUnlearn
    # Training setup
    dataset_retain: Literal['coco_object', 'coco_object_no_filter', 'imagenet243', 'imagenet243_no_filter'] = Field("coco_object", description='prompts corresponding to non-target concept to retain')
    adv_method: Literal['text_encoder', 'noxattn', 'selfattn', 'xattn', 'full', 'notime', 'xlayer', 'selflayer'] = Field(description='method of training', default='text_encoder')
    component: Literal['all', 'fc', 'attn'] = Field("all", description='component')
    norm_layer: Literal[True, False] = Field(False, description='During training, norm layer to be updated or not')
    adv_lr: Optional[float] = Field(1e-5, description='learning rate used to train')
    adv_iterations: Optional[int] = Field(1000, description='iterations used to train')
    adv_retain_batch: Optional[int] = Field(1, description='batch size of retaining prompt during training')
    adv_attack_embd_type: Literal['word_embd', 'condition_embd'] = Field("word_embd", description='the adversarial embd type: word embedding, condition embedding')
    adv_attack_type: Literal['replace_k' ,'add', 'prefix_k', 'suffix_k', 'mid_k', 'insert_k', 'per_k_words'] = Field("prefix_k", description='the attack type: append or add')
    adv_attack_init: Literal['random', 'latest'] = Field("latest", description='the attack init: random or latest')
    adv_attack_step: Optional[int] = Field(30, description='adversarial attack steps')
    adv_attack_lr: Optional[float] = Field(1e-3, description='learning rate used to train')
    adv_attack_method: Literal['pgd', 'multi_pgd', 'fast_at', 'free_at'] = Field('pgd', description='method of training')
    adv_retain_train: Literal['iter', 'reg'] = Field("iter", description='different retaining version: reg (regularization) or iter (iterative)')
    adv_retain_step: Optional[int] = Field(1, description='number of steps for retaining prompts')
    adv_retain_loss_w: Optional[float] = Field(1.0, description='retaining loss weight')
    # Attack hyperparameters
    adv_prompt_update_step: Optional[int] = Field(1, description='after every n step, adv prompt would be updated')
    adv_warmup_iter: Optional[int] = Field(200, description='the number of warmup interations before attack')
    adv_prompt_num: Optional[int] = Field(1, description='number of prompt token for adversarial soft prompt learning')
    

    # configs for LocoGen
    loco_concept_type: Optional[Literal["object", "style", "mem"]] = Field("object")
    eos: Optional[str] = Field('False', description= "If EOS tokens are used")
    # Regularization strength
    reg_key: Optional[float] = Field(default=0.01, description="Cuda operation")
    reg_value: Optional[float] = Field(0.01, description="Cuda operation")
    seq: Optional[int] = Field(4, description="Sequence length for operation")
    start_loc: Optional[int] = Field(8, description="Start location")


    # configs for UCE
    technique: Literal["replace", "tensor"] = Field('replace', description='technique to erase (either replace or tensor)')
    erase_scale: Optional[float] = Field(1.0, description='scale to erase concepts')


    # configs for MACE
    mace_lr: Optional[float] = Field(1e-5)
    mace_train_batch_size: Optional[int] = Field(1)
    mace_train_seperate: Literal[True, False] = Field(False)
    mace_dataloader_num_workers: Optional[int] = Field(0)
    mace_lr_warmup_steps: Optional[int] = Field(0)
    mace_lr_num_cycles: Optional[int] = Field(1)
    mace_lr_power: Optional[float] = Field(1.0)
    mace_max_train_steps: Optional[int] = Field(50) # it is set to 120 for explicit content.
    mace_importance_sampling: Literal[True, False] = Field(True)
    mace_num_train_epochs: Optional[int] = Field(1)
    use_gsam_mask: Literal[True, False] = Field(True)
    mace_rank: Optional[int] = Field(1)
    mace_lamb: Optional[float] = Field(1e+3)
    mace_train_preserve_scale: Optional[float] = Field(0.0)
    mace_preserve_weight: Optional[float] = Field(0.0)
    mace_max_memory: Optional[int] = Field(100)
    mace_concept_type: Optional[str] = Field("object")
    # making data for mace
    data_dir: Optional[str] = Field("mace-data")
    grounded_config: Optional[str] = Field("train_methods/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    grounded_checkpoint: Optional[str] = Field("train_methods/groundingdino_swint_ogc.pth")
    sam_hq_checkpoint: Optional[str] = Field("train_methods/sam_hq_vit_h.pth")

    
    # configs for receler
    receler_iterations: Optional[int] = Field(500, description='iterations used to train')
    receler_lr: Optional[float] = Field(3e-4, description='learning rate used to train')
    receler_concept_reg_weight: Optional[float] = Field(0.1, description='weight of concept-localized regularization loss')
    receler_mask_thres: Optional[float] = Field(0.1, description='threshold to obtain cross-attention mask')
    receler_advrs_iters: Optional[int] = Field(50, description='number of adversarial iterations')
    num_advrs_prompts: Optional[int] = Field(16, description='number of attack prompts to add')
    receler_rank: Optional[int] = Field(128, description='the rank of eraser')


    # configs for SDD (safe self-distill diffusion)
    sdd_method: Literal["full", "selfattn", "xattn", "noxattn", "notime"] = Field("xattn", desciption='method of training')
    sdd_num_steps: Optional[int] = Field(1300, description="The total number of training iterations to perform.")
    sdd_concept_method: Literal["composite", "random", "iterative", "sequential"] = Field(default="iterative")
    
    # configs for FMN (Forget-Me-Not)
    fmn_concept_type: Literal["object", "style", "naked"] = Field("object")
    fmn_train_batch_size: Optional[int] = Field(1)
    fmn_gradient_accumulation_steps: Optional[int] = Field(1)
    clip_ti_decay: Literal[True, False] = Field(True)
    fmn_lr_ti: Optional[float] = Field(0.001)
    fmn_scale_lr: Literal[True, False] = Field(True)
    fmn_max_train_steps_ti: Optional[int] = Field(500)
    fmn_weight_decay_ti: Optional[float] = Field(0.1)
    fmn_save_steps_ti: Optional[int] = Field(100)
    fmn_lr_warmup_steps_ti: Optional[int] = Field(100)
    fmn_lr_attn: Optional[float] = Field(2e-6)
    only_optimize_ca: Literal[True, False] = Field(False)
    use_pooler: Literal[True, False] = Field(True)
    center_crop: Literal[True, False] = Field(False)
    fmn_max_train_steps_attn: Optional[int] = Field(35)
    fmn_dataloader_num_workers: Optional[int] = Field(2)
    fmn_num_train_epochs: Optional[int] = Field(1)
    fmn_lr_power_attn: Optional[float] = Field(1.0)
    fmn_lr_num_cycles_attn: Optional[int] = Field(1)
    fmn_lr_warmup_steps_attn: Optional[int] = Field(0)
    instance_data_dir: Optional[str] = Field("fmn-data")


    # configs for SalUn
    salun_method: Optional[Literal["full", "selfattn", "xattn", "noxattn", "notime"]] = Field("xattn")
    salun_iter: Optional[int] = Field(1000)
    salun_lr_warmup_steps: Optional[int] = Field(500)
    salun_lr: Optional[float] = Field(1e-5)    
    salun_eta: Optional[float] = Field(0.0)
    # masking
    classes: Optional[str] = Field("1", description="erased class number in Imagenette. labels are [tench, English springer, cassette player, chainsaw, church, French horn, garbage truck, gas pump, golf ball parachute].")  # erasing class number 
    # ["tench", "English springer", "cassette player", "chain saw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
    salun_masking_batch_size: Optional[int] = Field(4)
    salun_masking_lr: Optional[float] = Field(1e-5)
    is_nsfw: Literal[True, False] = Field(False) 

    # configs for SPM
    # no configuration in SPM

    # config for DoCo
    doco_parameter_group: Literal["embedding", "cross-attn", "full-weight"] = Field("corss-attn") 
    doco_lr: Optional[float] = Field(6e-6)
    doco_dlr: Optional[float] = Field(1e-2)
    doco_batch_size: Optional[int] = Field(8)
    doco_concept_type: Optional[Literal["object", "style", "nudity", "violence"]] = Field("object")
    doco_num_class_images: Optional[int] = Field(1000)
    doco_num_class_prompts: Optional[int] = Field(200)
    doco_max_train_steps: Optional[int] = Field(2000)
    doco_num_train_epochs: Optional[int] = Field(1)
    doco_center_crop: Literal[True, False] = Field(False)
    doco_hflip: Literal[True, False] = Field(False)
    doco_noaug: Literal[True, False] = Field(False) # appropriate True when style erasing according to official implemantation
    doco_lr_warmup_steps: Optional[int] = Field(500)
    doco_dlr_warmup_steps: Optional[int] = Field(500, description="Number of steps for the warmup training of the discriminator.")
    doco_loss_type_reverse: Optional[str] = Field("model-based")
    doco_lambda_: Optional[float] = Field(1.0)

    # configs for AGE
    age_method: Literal["noxattn", "selfattn", "xattn", "xattn_matching", "full", "notime", "xlayer", "selflayer"] = Field("xattn")
    age_lr: Optional[float] = Field(1e-5)
    age_iters: Optional[int] = Field(1000)
    age_lamda: Optional[float] = Field(1.0)
    gumbel_topk: Optional[int] = Field(5, description="number of top-k values in the soft gumbel softmax to be considered")

    # configs for GLoCE
    gloce_method: Literal["unet_ca", "unet_ca_kv", "unet_ca_v", "unet_ca_ou,", "unet_sa_out","unet_sa", "unet_conv2d", "unet_misc", "te_attn"] = Field("unet_ca")
    gloce_start_timestep: Optional[int] = Field(10, description="Start timestep")
    gloce_end_timestep: Optional[int] = Field(20, description="End timestep")
    gloce_st_prompt_idx: Optional[int] = Field(-1)
    gloce_end_prompt_idx: Optional[int] = Field(-1)
    gloce_update_rank: Optional[int] = Field(-1)
    gloce_degen_rank: Optional[int] = Field(-1)
    gloce_gate_rank: Optional[int] = Field(-1)
    gloce_n_tokens: Optional[int] = Field(-1)
    gloce_eta: Optional[float] = Field(-1)
    gloce_lamb: Optional[float] = Field(-1)
    gloce_lamb2: Optional[float] = Field(-1)
    gloce_p_val: Optional[float] = Field(-1)
    gloce_last_layer: Optional[str] = Field("")
    gloce_opposite_for_map: Optional[bool] = Field(False)
    gloce_thresh: Optional[float] = Field(1.5)
    gloce_use_emb_cache: Optional[bool] = Field(True)
    gloce_param_cache_path: Optional[str] = Field("./importance_cache/org_comps/sd_v1.4", description="Path to parameter cache")
    gloce_emb_cache_path: Optional[str] = Field("./importance_cache/text_embs/sd_v1.4", description="Path to embedding cache")
    gloce_emb_cache_fn: Optional[str] = Field("text_emb_cache_w_sel_base_chris_evans_anchor5.pt", description="Embedding cache file name")
    gloce_buffer_path: Optional[str] = Field("./importance_cache/buffers")
    gloce_n_target_concepts: Optional[int] = Field(1, description="Number of target concepts")
    gloce_n_anchor_concepts: Optional[int] = Field(5, description="Number of anchor concepts")
    gloce_tar_concept_idx: Optional[int] = Field(0, description="Target concept index")
    gloce_delta: Optional[float] = Field(1e-5)
    gloce_alpha: Optional[float] = Field(1.0)
    gloce_replace_word: Literal["celeb", "artist", "explicit"] = Field("artist")
    gloce_prompts_file_target: Optional[str] = Field("captions/prompt_train_gloce_target.yaml")
    gloce_prompts_file_anchor: Optional[str] = Field("captions/prompt_train_gloce_anchor.yaml")
    gloce_prompts_file_update: Optional[str] = Field("captions/prompt_train_gloce_update.yaml")

    # configs for ACE
    ace_lr: Optional[float] = Field(1e-5)
    ace_iterations: Optional[int] = Field(1000)
    ace_surrogate_guidance_scale: Optional[float] = Field(3.0)
    ace_null_weight: Optional[float] = Field(0.8)
    ace_pr_weight: Optional[float] = Field(0.5)
    ace_pl_weight: Optional[float] = Field(0.5)
    ace_change_step_rate: Optional[float] = Field(1.0)
    ace_lora_rank: Optional[int] = Field(4)
    ace_anchor_batch_size: Optional[int] = Field(2)
    ace_surrogate_concept_clip_path: Optional[str] = Field(default=None)
    #example of ace_surrogate_concept_clip_path : "evaluation-outputs/cartoon_eval_test/SD3/evaluation_results_clip_CONCEPT_image_None.json"
    ace_anchor_prompt_path: Optional[str] = Field("data/concept_text/IP_character_concept.txt")

    # configs for STEREO
    stereo_method: Literal["noxattn", "xattn"] = Field("noxattn")
    stereo_mode: Literal["stereo", "attack", "both"] = Field("stereo")
    stereo_iteration: Optional[int] = Field(200)
    stereo_ste_lr: Optional[float] = Field(0.5e-5)
    stereo_reo_lr: Optional[float] = Field(2e-5)
    stereo_ci_lr: Optional[float] = Field(5e-3)
    stereo_ti_max_iters: Optional[int] = Field(3000, description="Maximum training steps for textual inversion")
    stereo_n_iters: Optional[int] = Field(4, description="Total number of erasure-attack iterations")
    stereo_compositional_guidance_scale: Optional[float] = Field(2.0, description="Compositional guidance scale. The value has to be +1 of the scale you would like to set. If the intended scale is 1.0, then the value has to be 2.0")
    stereo_initializer_token: Literal["person", "object", "art"] = Field("object")
    stereo_learnable_property: Literal["object", "style"] = Field("object")
    stereo_generic_prompt: Optional[str] = Field("a photo of a", description="Generic prompt for textual inversion visualization")
    stereo_num_of_adv_concepts: Optional[int] = Field(4, description="Number of adversarial concepts to use in REO")
    stereo_anchor_concept_path: Optional[str] = Field("captions/stereo_anchor_prompts.json", description="Path to anchor concept json used in REO stage")
    stereo_attack_eval_images: Optional[str] = Field("data/images/eval/nudity")


    # configs for AdaVD
    adavd_batch_size: Optional[int] = Field(10)
    adavd_total_timesteps: Optional[int] = Field(30)
    adavd_mode: Literal["original", "erase", "retain"] = Field("original")
    adavd_erase_type: Literal["object", "style", "celebrity"] = Field("object")
    adavd_sigmoid_a: Optional[float] = Field(100)
    adavd_sigmoid_b: Optional[float] = Field(0.93)
    adavd_sigmoid_c: Optional[float] = Field(2)
    adavd_record_type: Literal["keys", "values"] = Field("values")
    adavd_decomp_timestep: Optional[int] = Field(0)
    adavd_contents: Optional[str] = Field("")

    # configs for cpe
    cpe_network_rank: Optional[int] = Field(1)
    cpe_network_alpha: Optional[float] = Field(1.0)
    cpe_network_continual_rank: Optional[int] = Field(16)
    cpe_network_hidden_size: Optional[int] = Field(16)
    cpe_network_init_size: Optional[int] = Field(16)
    cpe_num_add_prompts: Optional[int] = Field(16)
    cpe_batch_size: Optional[int] = Field(1)
    cpe_iterations: Optional[int] = Field(450)
    cpe_lr: Optional[float] = Field(0.00003)
    cpe_lr_scheduler: Optional[str] = Field("cosine_with_restarts")
    cpe_lr_scheduler_num_cycles: Optional[int] = Field(1)
    cpe_lr_warmup_steps: Optional[int] = Field(5)
    cpe_num_stages: Optional[int] = Field(10)
    cpe_factor_init_iter: Optional[int] = Field(4)
    cpe_factor_init_lr: Optional[int] = Field(10)
    cpe_factor_init_lr_cycle: Optional[int] = Field(2)
    cpe_text_encoder_lr: Optional[float] = Field(1e-05)
    cpe_unet_lr: Optional[float] = Field(0.0001)
    cpe_adv_coef: Optional[float] = Field(1.0)
    cpe_pal: Optional[float] = Field(1e+4)
    cpe_do_adv_learn: Literal[True, False] = Field(True)
    cpe_adv_iters: Optional[int] = Field(450)
    cpe_adv_lr: Optional[float] = Field(0.01)
    cpe_replace_word: Optional[str] = Field("artist", description="abstract concept of the target's. For example, in the case of erasing Akira Toriyama, this concept is replaced into artist.")
    cpe_prompt_scripts_path: Optional[str] = Field("", description="path to template prompt file (in csv format). example is /train_artist/prompt_templates.csv")
    cpe_mixup: Literal[True, False] = Field(True)
    cpe_noise_scale: Optional[float] = Field(0.0)
    cpe_st_prompt_idx: Optional[int] = Field(1)
    cpe_end_prompt_idx: Optional[int] = Field(-1)
    cpe_skip_learned: Literal[True, False] = Field(False)
    cpe_gate_rank: Optional[int] = Field(16, description="same as cpe_network_continual_rank, cpe_network_hidden_size, and cpe_network_init_size.")

    # configs for RACE
    race_adv_train: Literal[True, False] = Field(False)
    race_esd_path: Optional[str] = Field("", description="pretrained esd model's path")
    race_lasso: Literal[True, False] = Field(False)
    race_adv_loss: Literal["l1", "l2"] = Field("l2")
    race_lr: Optional[float] = Field(1e-5)
    race_iterations: Optional[int] = Field(1000)
    race_epsilon: Optional[float] = Field(0.1)
    race_pgd_num_step: Optional[int] = Field(10)

    # inference part
    prompt: Optional[str] = Field("a photo of the English springer", description="prompt in inference phase")
    negative_prompt: Optional[str] = Field("")
    images_dir: Optional[str] = Field("gen-images")
    erased_model_dir: Optional[str] = Field("models")
    guidance_scale: Optional[float] = Field(7.5, description="CFG scale")
    num_images_per_prompt: Optional[int] = Field(5)
    num_inference_steps: Optional[float] = Field(30)
    matching_metric: Optional[str] = Field("clipcos_tokenuni", description="matching metric for prompt vs erased concept")
    
    
    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()
        fields = cls.model_fields
        for name, field in fields.items():
            parser.add_argument(f"--{name}", default=field.default, help=field.description)
        return cls.model_validate(parser.parse_args().__dict__)
