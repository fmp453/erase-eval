import argparse
import importlib

from typing import Literal
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
    method: Literal["esd", "ac", "eap", "adv", "locogen", "uce", "mace", "receler", "fmn", "salun", "spm", "sdd", "diffquickfix", "doco", "gloce", "age", "ant", "ef", "mce", "original"] = Field("esd")
    sd_version: str = Field("compvis/stable-diffusion-v1-4")
    device: str = Field("0", description="gpu id. when using two gpus, separated by comma")
    seed: int = Field(0)

    # training part
    # general configs
    concepts: str = Field("English springer", description="separated by comma")
    save_dir: str = Field("models", description="path to dir for erased models")

    anchor_concept: str = Field("dog")
    seperator: str | None = Field(None, description='separator if you want to train bunch of erased_words separately')

    image_size: int = Field(512, description='image size used to train')
    ddim_steps: int = Field(50, description='ddim steps of inference used to train')
    ddpm_steps: int = Field(1000)
    max_grad_norm: float = Field(1.0)
    lr_scheduler: Literal["constant", "linear","cosine", "cosine_warmup", "cosine_warmup_restart", "polynomial", "polynomial_warmup", "polynomial_warmup_restart"] = Field("constant", description="learning rate scheduler")
    
    negative_guidance: float = Field(1, description='guidance of negative training used to train')
    start_guidance: float = Field(3, description='guidance of start image used to train')

    # for EAP and AGE
    gumbel_lr: float = Field(1e-3, description='learning rate for prompt')
    gumbel_temp: float = Field(2, description='temperature for gumbel softmax')
    gumbel_hard: Literal[0, 1] = Field(0, description='hard for gumbel softmax, 0: soft, 1: hard')
    gumbel_num_centers: int = Field(100, description='number of centers for kmeans, if <= 0 then do not apply kmeans')
    gumbel_update: int = Field(100, description='update frequency for preserved set, if <= 0 then do not update')
    gumbel_time_step: int = Field(0, description='time step for the starting point to estimate epsilon')
    gumbel_multi_steps: int = Field(2, description='multi steps for calculating the output')
    gumbel_k_closest: int = Field(1000, description='number of closest tokens to consider')
    ignore_special_tokens: bool = Field(True, description='ignore special tokens in the embedding matrix')
    vocab: str = Field("EN3K", description='vocab')
    pgd_num_steps: int = Field(2, description='number of step to optimize adversarial concepts')

    # configs for ESD (Erased Stable Diffusion)
    esd_method: Literal["full", "selfattn", "xattn", "noxattn", "notime"] = Field("xattn", description="which parameters are updated")
    esd_iter: int = Field(1000)
    esd_lr: float = Field(1e-5)
    esd_eta: float = Field(0.0)
    esd_lr_warmup_steps: int = Field(500)
    

    # configs for AC (Ablating Concepts)
    ac_method: Literal["full", "xattn"] = Field("xattn", description="which parameters are updated")
    ac_lr: float = Field(2e-6)
    
    ac_img_dir: str = Field("images")
    ac_prompt_path: str = Field("dog.csv")
    ac_concept_type: Literal["object", "style", "mem"] = Field("object")
    ac_batch_size: int = Field(8)
    

    # configs for EAP (Erasing-Adversarial-Preservation)
    eap_method: Literal["full", "selfattn", "xattn", "noxattn", "notime", "xattn_matching", "xlayer", "selflayer"] = Field("xattn", description='method of training')
    eap_iterations: int = Field(1000, description='iterations used to train')
    eap_lr: float = Field(1e-5, description='learning rate used to train')

    # configs for AdvUnlearn
    # Training setup
    dataset_retain: Literal['coco_object', 'coco_object_no_filter', 'imagenet243', 'imagenet243_no_filter'] = Field("coco_object", description='prompts corresponding to non-target concept to retain')
    adv_method: Literal['text_encoder', 'noxattn', 'selfattn', 'xattn', 'full', 'notime', 'xlayer', 'selflayer'] = Field('text_encoder', description='method of training')
    component: Literal['all', 'fc', 'attn'] = Field("all", description='component')
    norm_layer: bool = Field(False, description='During training, norm layer to be updated or not')
    adv_lr: float = Field(1e-5, description='learning rate used to train')
    adv_iterations: int = Field(1000, description='iterations used to train')
    adv_retain_batch: int = Field(1, description='batch size of retaining prompt during training')
    adv_attack_embd_type: Literal['word_embd', 'condition_embd'] = Field("word_embd", description='the adversarial embd type: word embedding, condition embedding')
    adv_attack_type: Literal['replace_k' ,'add', 'prefix_k', 'suffix_k', 'mid_k', 'insert_k', 'per_k_words'] = Field("prefix_k", description='the attack type: append or add')
    adv_attack_init: Literal['random', 'latest'] = Field("latest", description='the attack init: random or latest')
    adv_attack_step: int = Field(30, description='adversarial attack steps')
    adv_attack_lr: float = Field(1e-3, description='learning rate used to train')
    adv_attack_method: Literal['pgd', 'multi_pgd', 'fast_at', 'free_at'] = Field('pgd', description='method of training')
    adv_retain_train: Literal['iter', 'reg'] = Field("iter", description='different retaining version: reg (regularization) or iter (iterative)')
    adv_retain_step: int = Field(1, description='number of steps for retaining prompts')
    adv_retain_loss_w: float = Field(1.0, description='retaining loss weight')
    # Attack hyperparameters
    adv_prompt_update_step: int = Field(1, description='after every n step, adv prompt would be updated')
    adv_warmup_iter: int = Field(200, description='the number of warmup interations before attack')
    adv_prompt_num: int = Field(1, description='number of prompt token for adversarial soft prompt learning')
    

    # configs for LocoGen
    loco_concept_type: Literal["object", "style", "mem"] = Field("object")
    eos: str = Field('False', description= "If EOS tokens are used")
    # Regularization strength
    reg_key: float = Field(0.01, description="Cuda operation")
    reg_value: float = Field(0.01, description="Cuda operation")
    seq: int = Field(4, description="Sequence length for operation")
    start_loc: int = Field(8, description="Start location")


    # configs for UCE
    technique: Literal["replace", "tensor"] = Field('replace', description='technique to erase (either replace or tensor)')
    erase_scale: float = Field(1.0, description='scale to erase concepts')


    # configs for MACE
    mace_lr: float = Field(1e-5)
    mace_train_batch_size: int = Field(1)
    mace_train_seperate: bool = Field(False)
    mace_dataloader_num_workers: int = Field(0)
    mace_lr_warmup_steps: int = Field(0)
    mace_lr_num_cycles: int = Field(1)
    mace_lr_power: float = Field(1.0)
    mace_max_train_steps: int = Field(50) # it is set to 120 for explicit content.
    mace_importance_sampling: bool = Field(True)
    mace_num_train_epochs: int = Field(1)
    use_gsam_mask: bool = Field(True)
    mace_rank: int = Field(1)
    mace_lamb: float = Field(1e+3)
    mace_train_preserve_scale: float = Field(0.0)
    mace_preserve_weight: float = Field(0.0)
    mace_max_memory: int = Field(100)
    mace_concept_type: str = Field("object")
    # making data for mace
    data_dir: str = Field("mace-data")
    grounded_config: str = Field("train_methods/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    grounded_checkpoint: str = Field("train_methods/groundingdino_swint_ogc.pth")
    sam_hq_checkpoint: str = Field("train_methods/sam_hq_vit_h.pth")

    
    # configs for receler
    receler_iterations: int = Field(500, description='iterations used to train')
    receler_lr: float = Field(3e-4, description='learning rate used to train')
    receler_concept_reg_weight: float = Field(0.1, description='weight of concept-localized regularization loss')
    receler_mask_thres: float = Field(0.1, description='threshold to obtain cross-attention mask')
    receler_advrs_iters: int = Field(50, description='number of adversarial iterations')
    num_advrs_prompts: int = Field(16, description='number of attack prompts to add')
    receler_rank: int = Field(128, description='the rank of eraser')


    # configs for SDD (safe self-distill diffusion)
    sdd_method: Literal["full", "selfattn", "xattn", "noxattn", "notime"] = Field("xattn", description='method of training')
    sdd_num_steps: int = Field(1300, description="The total number of training iterations to perform.")
    sdd_concept_method: Literal["composite", "random", "iterative", "sequential"] = Field("iterative")
    
    # configs for FMN (Forget-Me-Not)
    fmn_concept_type: Literal["object", "style", "naked"] = Field("object")
    fmn_train_batch_size: int = Field(1)
    fmn_gradient_accumulation_steps: int = Field(1)
    clip_ti_decay: bool = Field(True)
    fmn_lr_ti: float = Field(0.001)
    fmn_scale_lr: bool = Field(True)
    fmn_max_train_steps_ti: int = Field(500)
    fmn_weight_decay_ti: float = Field(0.1)
    fmn_save_steps_ti: int = Field(100)
    fmn_lr_warmup_steps_ti: int = Field(100)
    fmn_lr_attn: float = Field(2e-6)
    only_optimize_ca: bool = Field(False)
    use_pooler: bool = Field(True)
    center_crop: bool = Field(False)
    fmn_max_train_steps_attn: int = Field(35)
    fmn_dataloader_num_workers: int = Field(2)
    fmn_num_train_epochs: int = Field(1)
    fmn_lr_power_attn: float = Field(1.0)
    fmn_lr_num_cycles_attn: int = Field(1)
    fmn_lr_warmup_steps_attn: int = Field(0)
    instance_data_dir: str = Field("fmn-data")


    # configs for SalUn
    salun_method: Literal["full", "selfattn", "xattn", "noxattn", "notime"] = Field("xattn")
    salun_iter: int = Field(1000)
    salun_lr_warmup_steps: int = Field(500)
    salun_lr: float = Field(1e-5)    
    salun_eta: float = Field(0.0)
    # masking
    classes: str = Field("1", description="erased class number in Imagenette. labels are [tench, English springer, cassette player, chainsaw, church, French horn, garbage truck, gas pump, golf ball parachute].")  # erasing class number 
    # ["tench", "English springer", "cassette player", "chain saw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
    salun_masking_batch_size: int = Field(4)
    salun_masking_lr: float = Field(1e-5)
    is_nsfw: bool = Field(False) 

    # configs for SPM
    # no configuration in SPM

    # config for DoCo
    doco_parameter_group: Literal["embedding", "cross-attn", "full-weight"] = Field("cross-attn") 
    doco_lr: float = Field(6e-6)
    doco_dlr: float = Field(1e-2)
    doco_batch_size: int = Field(8)
    doco_concept_type: Literal["object", "style", "nudity", "violence"] = Field("object")
    doco_num_class_images: int = Field(1000)
    doco_num_class_prompts: int = Field(200)
    doco_max_train_steps: int = Field(2000)
    doco_num_train_epochs: int = Field(1)
    doco_center_crop: bool = Field(False)
    doco_hflip: bool = Field(False)
    doco_noaug: bool = Field(False) # appropriate True when style erasing according to official implemantation
    doco_lr_warmup_steps: int = Field(500)
    doco_dlr_warmup_steps: int = Field(500, description="Number of steps for the warmup training of the discriminator.")
    doco_loss_type_reverse: str = Field("model-based")
    doco_lambda_: float = Field(1.0)

    # configs for AGE
    age_method: Literal["noxattn", "selfattn", "xattn", "xattn_matching", "full", "notime", "xlayer", "selflayer"] = Field("xattn")
    age_lr: float = Field(1e-5)
    age_iters: int = Field(1000)
    age_lamda: float = Field(1.0)
    gumbel_topk: int = Field(5, description="number of top-k values in the soft gumbel softmax to be considered")

    # configs for GLoCE
    gloce_method: Literal["unet_ca", "unet_ca_kv", "unet_ca_v", "unet_ca_ou,", "unet_sa_out","unet_sa", "unet_conv2d", "unet_misc", "te_attn"] = Field("unet_ca")
    gloce_start_timestep: int = Field(10, description="Start timestep")
    gloce_end_timestep: int = Field(20, description="End timestep")
    gloce_st_prompt_idx: int = Field(-1)
    gloce_end_prompt_idx: int = Field(-1)
    gloce_update_rank: int = Field(-1)
    gloce_degen_rank: int = Field(-1)
    gloce_gate_rank: int = Field(-1)
    gloce_n_tokens: int = Field(-1)
    gloce_eta: float = Field(-1)
    gloce_lamb: float = Field(-1)
    gloce_lamb2: float = Field(-1)
    gloce_p_val: float = Field(-1)
    gloce_last_layer: str = Field("")
    gloce_opposite_for_map: bool = Field(False)
    gloce_thresh: float = Field(1.5)
    gloce_use_emb_cache: bool = Field(True)
    gloce_param_cache_path: str = Field("./importance_cache/org_comps/sd_v1.4", description="Path to parameter cache")
    gloce_emb_cache_path: str = Field("./importance_cache/text_embs/sd_v1.4", description="Path to embedding cache")
    gloce_emb_cache_fn: str = Field("text_emb_cache_w_sel_base_chris_evans_anchor5.pt", description="Embedding cache file name")
    gloce_buffer_path: str = Field("./importance_cache/buffers")
    gloce_n_target_concepts: int = Field(1, description="Number of target concepts")
    gloce_n_anchor_concepts: int = Field(5, description="Number of anchor concepts")
    gloce_tar_concept_idx: int = Field(0, description="Target concept index")
    gloce_delta: float = Field(1e-5)
    gloce_alpha: float = Field(1.0)
    gloce_replace_word: Literal["celeb", "artist", "explicit"] = Field("artist")
    gloce_prompts_file_target: str = Field("captions/prompt_train_gloce_target.yaml")
    gloce_prompts_file_anchor: str = Field("captions/prompt_train_gloce_anchor.yaml")
    gloce_prompts_file_update: str = Field("captions/prompt_train_gloce_update.yaml")

    # configs for ACE
    ace_lr: float = Field(1e-5)
    ace_iterations: int = Field(1000)
    ace_surrogate_guidance_scale: float = Field(3.0)
    ace_null_weight: float = Field(0.8)
    ace_pr_weight: float = Field(0.5)
    ace_pl_weight: float = Field(0.5)
    ace_change_step_rate: float = Field(1.0)
    ace_lora_rank: int = Field(4)
    ace_anchor_batch_size: int = Field(2)
    ace_surrogate_concept_clip_path: str | None = Field(None)
    #example of ace_surrogate_concept_clip_path : "evaluation-outputs/cartoon_eval_test/SD3/evaluation_results_clip_CONCEPT_image_None.json"
    ace_anchor_prompt_path: str = Field("data/concept_text/IP_character_concept.txt")

    # configs for STEREO
    stereo_method: Literal["noxattn", "xattn"] = Field("noxattn")
    stereo_mode: Literal["stereo", "attack", "both"] = Field("stereo")
    stereo_iteration: int = Field(200)
    stereo_ste_lr: float = Field(0.5e-5)
    stereo_reo_lr: float = Field(2e-5)
    stereo_ci_lr: float = Field(5e-3)
    stereo_ti_max_iters: int = Field(3000, description="Maximum training steps for textual inversion")
    stereo_n_iters: int = Field(4, description="Total number of erasure-attack iterations")
    stereo_compositional_guidance_scale: float = Field(2.0, description="Compositional guidance scale. The value has to be +1 of the scale you would like to set. If the intended scale is 1.0, then the value has to be 2.0")
    stereo_initializer_token: Literal["person", "object", "art"] = Field("object")
    stereo_learnable_property: Literal["object", "style"] = Field("object")
    stereo_generic_prompt: str = Field("a photo of a", description="Generic prompt for textual inversion visualization")
    stereo_num_of_adv_concepts: int = Field(4, description="Number of adversarial concepts to use in REO")
    stereo_anchor_concept_path: str = Field("captions/stereo_anchor_prompts.json", description="Path to anchor concept json used in REO stage")
    stereo_attack_eval_images: str = Field("data/images/eval/nudity")


    # configs for AdaVD
    adavd_batch_size: int = Field(10)
    adavd_total_timesteps: int = Field(30)
    adavd_mode: Literal["original", "erase", "retain"] = Field("original")
    adavd_erase_type: Literal["object", "style", "celebrity"] = Field("object")
    adavd_sigmoid_a: float = Field(100)
    adavd_sigmoid_b: float = Field(0.93)
    adavd_sigmoid_c: float = Field(2)
    adavd_record_type: Literal["keys", "values"] = Field("values")
    adavd_decomp_timestep: int = Field(0)
    adavd_contents: str = Field("")

    # configs for cpe
    cpe_network_rank: int = Field(1)
    cpe_network_alpha: float = Field(1.0)
    cpe_network_continual_rank: int = Field(16)
    cpe_network_hidden_size: int = Field(16)
    cpe_network_init_size: int = Field(16)
    cpe_num_add_prompts: int = Field(16)
    cpe_batch_size: int = Field(1)
    cpe_iterations: int = Field(450)
    cpe_lr: float = Field(0.00003)
    cpe_lr_scheduler: str = Field("cosine_with_restarts")
    cpe_lr_scheduler_num_cycles: int = Field(1)
    cpe_lr_warmup_steps: int = Field(5)
    cpe_num_stages: int = Field(10)
    cpe_factor_init_iter: int = Field(4)
    cpe_factor_init_lr: int = Field(10)
    cpe_factor_init_lr_cycle: int = Field(2)
    cpe_text_encoder_lr: float = Field(1e-05)
    cpe_unet_lr: float = Field(0.0001)
    cpe_adv_coef: float = Field(1.0)
    cpe_pal: float = Field(1e+4)
    cpe_do_adv_learn: bool = Field(True)
    cpe_adv_iters: int = Field(450)
    cpe_adv_lr: float = Field(0.01)
    cpe_replace_word: str = Field("artist", description="abstract concept of the target's. For example, in the case of erasing Akira Toriyama, this concept is replaced into artist.")
    cpe_prompt_scripts_path: str = Field("", description="path to template prompt file (in csv format). example is /train_artist/prompt_templates.csv")
    cpe_mixup: bool = Field(True)
    cpe_noise_scale: float = Field(0.0)
    cpe_st_prompt_idx: int = Field(1)
    cpe_end_prompt_idx: int = Field(-1)
    cpe_skip_learned: bool = Field(False)
    cpe_gate_rank: int = Field(16, description="same as cpe_network_continual_rank, cpe_network_hidden_size, and cpe_network_init_size.")

    # configs for RACE
    race_adv_train: bool = Field(False)
    race_esd_path: str = Field("", description="pretrained esd model's path")
    race_lasso: bool = Field(False)
    race_adv_loss: Literal["l1", "l2"] = Field("l2")
    race_lr: float = Field(1e-5)
    race_iterations: int = Field(1000)
    race_epsilon: float = Field(0.1)
    race_pgd_num_step: int = Field(10)

    # configs for ant
    ant_method: Literal["full", "selfattn", "xattn", "noxattn", "notime", "xattn_matching", "xlayer", "selflayer"] = Field("full", description='method of training')
    ant_iterations: int = Field(250)
    ant_lr: float = Field(5e-4)
    ant_before_step: int = Field(7)
    ant_alpha_1: float = Field(1.0)
    ant_alpha_2: float = Field(0.5)
    ant_mask_path: str | None = Field(None)
    ant_if_gradient: bool = Field(True)

    # configs for EraseFlow
    ef_use_8bit_adam: bool = Field(False)
    ef_lr: float = Field(3e-4)
    ef_flow_lr: float = Field(3e-4)
    ef_adam_beta1: float = Field(0.9)
    ef_adam_beta2: float = Field(0.999)
    ef_adam_weight_decay: float = Field(0.01)
    ef_adam_epsilon: float = Field(1e-8)
    ef_eta: float = Field(1.0)
    ef_logbeta: float = Field(2.5)
    ef_batch_size: int = Field(1)
    ef_lora_rank: int = Field(4)
    ef_switch_epoch: int = Field(20)
    ef_num_epochs: int = Field(20)

    # configs for MCE
    # mce.data
    mce_metadata: str = Field("datasets/gcc3m/Validation_GCC-1.1.0-Validation.tsv")
    mce_deconceptmeta: str = Field("configs/concept_long.yaml", description="need to merge all the concepts in one config file")
    mce_only_deconcept_latent: bool = Field(True)
    mce_size: int = Field(40)
    mce_batch_size: int = Field(1)
    mce_style: Literal["concept", "style", "nsfw"] = Field("concept")
    mce_filter_ratio: float = Field(0.9)
    mce_with_fg_filter: bool = Field(False)
    mce_with_synonyms: bool = Field(False)

    # mce.trainer
    mce_epochs: int = Field(5)
    mce_beta: float = Field(0.1)
    mce_epsilon: float = Field(0.0)
    mce_lr: float = Field(0.5)
    mce_attn_lr: float = Field(0)
    mce_ff_lr: float = Field(0.5)
    mce_n_lr: float = Field(0.5)
    mce_model: Literal["sd1", "sd2", "sdxl", "sd3", "flux", "dit"] = Field("flux")
    mce_num_intervention_steps: int = Field(5)
    mce_init_lambda: int = Field(3)
    mce_regex: Literal[".*", "^(down_blocks).*", "^(up_blocks).*"] = Field(".*", description="^(down_blocks.[1,2]).* optional are ^(down_blocks).*, ^(up_blocks).*, .* (for all heads)")
    mce_attn_name: str = Field("attn", description="use to filter the attention heads, e.g. attn2 only for cross attention")
    mce_head_num_filter: int = Field(1, description="number of heads to filter, apply lambda to the layter that has more than head_num_filter heads")
    mce_masking: Literal["sigmoid", "hard_discrete"] = Field("hard_discrete" )
    mce_masking_eps: float = Field(0.5)
    mce_disable_progress_bar: bool = Field(True)
    mce_accumulate_grad_batches: int = Field(4)
    mce_grad_checkpointing: bool = Field(True)

    # mce.lr_scheduler
    mce_lr_warmup_steps: int = Field(10)
    mce_lr_num_cycles: int = Field(1)
    mce_lr_power: float = Field(1.0)
    mce_lr_decay_steps: int = Field(0)

    # mce.loss
    mce_reg: Literal[0, 1, 2] = Field(1, description="2 for L2 norm, 1 for L1 norm, 0 for L0 norm")
    mce_reconstruct: Literal[1, 2] = Field(2, description="2 for L2 norm, 1 for L1 norm")
    mce_mean: bool = Field(True)
    mce_use_attn_reg: bool = Field(True)
    mce_use_ffn_reg: bool = Field(True)
    mce_lambda_reg: bool = Field(True)
    mce_reg_alpha: float = Field(0.4)
    mce_reg_beta: int = Field(1, description="no need to use beta for now for testing")

    # config for CoGFD
    cogfd_p1: float = Field(-1.0)
    cogfd_p2: float = Field(1.0)
    cogfd_start: int = Field(990)
    cogfd_end: int = Field(1000)
    cogfd_lr: float = Field(5e-5)
    cogfd_num_train_epochs: int = Field(1)
    cogfd_train_batch_size: int = Field(20)
    cogfd_adam_beta_1: float = Field(0.9)
    cogfd_adam_beta_2: float = Field(0.999)
    cogfd_adam_weight_decay: float = Field(0.01)
    cogfd_adam_epsilon: float = Field(1.0e-08)
    cogfd_gradient_accumulation_steps: int = Field(1)
    cogfd_scale_lr: bool = Field(False)
    cogfd_use_8bit_adam: bool = Field(False)
    cogfd_train_text_encoder: bool = Field(False)
    cogfd_center_crop: bool = Field(False)
    cogfd_only_optimize_ca: bool = Field(False)
    cogfd_set_grads_to_none: bool = Field(False)
    cogfd_use_pooler: bool = Field(True)
    cogfd_max_train_steps: int = Field(100)
    cogfd_lr_warmup_steps: int = Field(0)
    cogfd_lr_num_cycles: int = Field(1)
    cogfd_lr_power: float = Field(1.0)
    cogfd_dataloader_num_workers: int = Field(9)
    cogfd_graph_path: str = Field("cpgfd-graph/graph.json")
    cogfd_iterate_n: int = Field(2)
    cogfd_combine_concept_x: str = Field("A child is drinking wine")
    cogfd_combine_theme_y: str = Field("underage drinking")


    # inference part
    prompt: str = Field("a photo of the English springer", description="prompt in inference phase")
    negative_prompt: str = Field("")
    images_dir: str = Field("gen-images")
    erased_model_dir: str = Field("models")
    guidance_scale: float = Field(7.5, description="CFG scale")
    num_images_per_prompt: int = Field(5)
    num_inference_steps: float = Field(30)
    matching_metric: str = Field("clipcos_tokenuni", description="matching metric for prompt vs erased concept")
    
    
    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()
        fields = cls.model_fields
        for name, field in fields.items():
            annotation = field.annotation
            default = field.default
            help_text = field.description
            if annotation is bool:
                if default is False:
                    parser.add_argument(
                        f"--{name}",
                        action="store_true",
                        help=help_text
                    )
                else:
                    parser.add_argument(
                        f"--{name}",
                        action="store_false",
                        help=help_text
                    )
            else:
                parser.add_argument(f"--{name}", default=field.default, help=field.description)
            parser.add_argument(f"--{name}", default=field.default, help=field.description)
        return cls.model_validate(vars(parser.parse_args()))
