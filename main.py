# Residual Local Feature Network for Efficient Super-Resolution
# Residual in Residual Dense Block
# https://mmagic.readthedocs.io/en/latest/_modules/mmagic/models/editors/ddpm/denoising_unet.html
# https://huggingface.co/eugenesiow/edsr-base
# https://colab.research.google.com/github/eugenesiow/super-image-notebooks/blob/master/notebooks/Train_super_image_Models.ipynb#scrollTo=YTQpnw8dA_Sf
# https://www.kaggle.com/datasets/sharansmenon/div2k/code
import torch

from trainer import eval_trained_models

if __name__ == '__main__':
    torch.cuda.empty_cache()
    eval_trained_models()
