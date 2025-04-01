# Erasing with Precision: Evaluating Specific Concept Erasure from Text-to-Image Generative Models

Official implementation of the paper, [Erasing with Precision: Evaluating Specific Concept Erasure from Text-to-Image Generative Models](https://arxiv.org/abs/2502.13989).

## Install (our experiments)
1. base environments
    ```bash
    docker pull pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
    ```

2. other dependencies (in the container)
    ```bash
    apt update && apt upgrade
    apt install git
    apt install wget
    pip install -r requirements.txt
    wget -P train_methods https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    wget -P train_methods https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth
    ```

## Install (in general)
```bash
apt update && apt upgrade
apt install git
apt install wget 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install  -r requirements.txt
wget -P train_methods https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P train_methods https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth
```

## Erasing
```bash
python main.py --mode train --method esd --concepts "English springer" --device "0,1"
```

For Ablating Concepts, make `.csv` file contains the over 200 prompts including guided concept.
```bash
python main.py --mode train --method ac --concepts "English springer" --device "0,1" --ac_prompt_path dog.csv
```

For more arguments can be shown in `utils.py`.

## Inference
```bash
python main.py --mode infer --method esd --prompt "a photo of English springer" --erased_model $MODEL_DIR
```

## Evaluation
Before the evaluation, you need to get openai api key and huggingface token.
```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```


```bash
python eval.py --method esd --protocol 1 --device 0
```

## Citation
Our paper can be cited as follows. 
```tex
@misc{fuchi2025erasingprecisionevaluatingspecific,
      title={Erasing with Precision: Evaluating Specific Concept Erasure from Text-to-Image Generative Models}, 
      author={Masane Fuchi and Tomohiro Takagi},
      year={2025},
      eprint={2502.13989},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.13989}, 
}
```

## Acknowledgement
We reimplemented several methods using [🤗 Diffusers](https://github.com/huggingface/diffusers)

- [Erased Stable Diffusion (ESD)](https://github.com/rohitgandikota/erasing)
- [Ablating Concepts (AC)](https://github.com/nupurkmr9/concept-ablation)
- [Unified Concept Editing (UCE)](https://github.com/rohitgandikota/unified-concept-editing)
- [Semi-Permeable Membrane (SPM)](https://github.com/Con6924/SPM)
- [Safe self-Distillation Diffusion (SDD)](https://github.com/nannullna/safe-diffusion)
- [Mass Concept Erasure (MACE)](https://github.com/Shilin-LU/MACE)
- [Reliable Concept Erasing via Lightweight Erasers (Receler)](https://github.com/jasper0314-huang/Receler)
- [Erasing-Adversarial-Preservation (EAP)](https://github.com/tuananhbui89/Erasing-Adversarial-Preservation)
- [LocoEdit](https://github.com/samyadeepbasu/LocoGen)
- [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency)
- [Forget-Me-Not (FMN)](https://github.com/SHI-Labs/Forget-Me-Not)

File Stracture is followed by [TabSyn](https://github.com/amazon-science/tabsyn). 

MSCOCO provided by https://huggingface.co/datasets/shunk031/MSCOCO

CMMD PyTorch Source Code: https://github.com/sayakpaul/cmmd-pytorch 
