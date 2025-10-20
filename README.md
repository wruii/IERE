# IERE

# Introduction
Official code of the paper "Boosting Cross-Domain Semi-supervised Medical Image Segmentation with Internal and External Regularizations"

# Usage
We provide code and model for Prostate dataset.
Model could be got at [pretrain_model](https://pan.quark.cn/s/90b512bbd4ba)

To train a model,
```python
python train_sam_prostate_cp.py # for Prostate training from domain B to domain A
python train_sam_prostate_cp_ab.py # for Prostate training from domain A to domain B
```

To test a model,
```python
python inference_prostate.py # for Prostate test
```
## Acknowledgements
Our code is largely based on [SSL-MedSeg](https://github.com/kaland313/SSL-MedSeg.git) and [BCP](https://github.com/DeepMed-Lab-ECNU/BCP.git). Thanks for these authors' valuable work. 
