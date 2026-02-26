# SAD-GDD
The official PyTorch implementation for the following paper:Supervised Anomaly Detection for Generalized Deepfake Detection.

## 1. Datasets
- Generate FF_train.csv, FF_val.csv, FF_test.csv according to the following format:
```
For example:
  |- FF_train.csv
  	|_ img_path,label,
  		/path of cropped_images/imgxx.png, 1,
  		...
```


- Video-level AUCs are reported for multiple datasets. Training is conducted on the FF++ (c23).
  
|Datasets|FF++|CD2|DFDC|DFDCP|DF1.0 std/rand|
|-|-|-|-|-|-|
|AUC (%)|98.76|92.02|78.21|86.81|94.01|


## 2. Pretrained model
Before running the training code, make sure you load the pre-trained weights. We provide pre-trained weights under [./models/pretrained](). You can also download Swin-Base model trained on ImageNet-22K (through this [link](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)).
## 3. Train
For model training, we provide a sh file to train our model by running sh train.sh.
```
For example:
srun -p normal -w cluster -c 20 --mem=80G --gres=gpu:3080ti:4  -u sh train.sh 
```
## 4. Test
For model testing, we provide a sh file to test our model by running sh predict.sh.
```
For example:
srun -p normal -w cluster-2 -c 8 --mem=20G --gres=gpu:3080ti:1 -u sh predict.sh
```

## Citation
Please kindly consider citing our papers in your publications.
```
@inproceedings{Lu2026_supervise,
    title={Supervised Anomaly Detection for Generalized Deepfake Detection},
    author={Lin, Lu and Guangshuai Gao},
    journal={IEEE Signal Processing Letters},
    volume={},
    pages={},
    year={2026},
    publisher={IEEE}
}
```
