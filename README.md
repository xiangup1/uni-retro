# uni-retro



## Platform

[Uni-Retro platform](https://app.bohrium.dp.tech/retro-synthesis/workbench/): A multi-step retrosynthesis platform that integrates the uni-retro.

## Environment Setup

To begin working with uni-retro, you'll need to set up your environment. Below is a step-by-step guide to get you started:

```bash
# Install Uni-Core
git clone https://github.com/dptech-corp/Uni-Core
cd Uni-Core
pip install .
cd -

# Install Unimol V2 DEV
cd unimol_v2_dev
pip install .
cd -

# Install additional dependencies
pip install -r requirements.txt

```

## Datasets and Pretrained Weights

You can obtain the model from [the Google Drive](https://drive.google.com/drive/folders/1lZOLRGyZy18EVow7gyxtKWvs_yuwlIE3?usp=sharing):


## Model Validation

To validate the NAG2G model with the provided weights, follow the instructions below:

```bash
# Execute retrosynthesis pipeline for target smiles
sh infer.sh CCOC(=O)c1ccc(Nc2ncn(-c3ccnc(N4CC(C)N(C(C)=O)C(C)C4)c3)n2)cc1 
```

For any questions or issues, please open an issue on our GitHub repository.

Thank you for your interest in uni-retro!
