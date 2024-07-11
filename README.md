![block](./images/title.gif)

## 🚀 Get Started
**Installation**
Install [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Shap-E](https://github.com/openai/shap-e#usage) as fellow:
```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install ninja
pip install -r requirements.txt

git clone https://github.com/hustvl/GaussianDreamer.git 
cd GaussianDreamer

pip install ./gaussiansplatting/submodules/diff-gaussian-rasterization
pip install ./gaussiansplatting/submodules/simple-knn

git clone https://github.com/openai/shap-e.git
cd shap-e
pip install -e .
```
Download [finetuned Shap-E](https://huggingface.co/datasets/tiange/Cap3D/tree/main/our_finetuned_models) by Cap3D, and put it in `./load`

**Quickstart**

Text-to-3D Generation
```
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a fox"

# if you want to import the generated 3D assets into the Unity game engine.
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a fox" system.sh_degree=3 
```

Text-to-Avatar Generation
```
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Spiderman stands with open arms" system.load_type=1

# if you want to import the generated 3D assets into the Unity game engine.
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Spiderman stands with open arms" system.load_type=1 system.sh_degree=3 
```
