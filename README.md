# SAD-GDD
The official PyTorch implementation for the following paper:Supervised Anomaly Detection for Generalized Deepfake Detection

## 1. Datasets
- Generate FF_train.csv, FF_val.csv, FF_test.csv according to the following format:
```
  |- FF_train.csv
  	|_ img_path,label,
  		/path/to/cropped_images/imgxx.png, 1,
  		...
```


1. listee
22. lista
3. list

   [link](https://github.com/lulin60/SAD-GDD/edit/main/README.md#L4C2)

![image_name]()

|col1|col2|col3|
|-|-|-|
|data1|data2|data3|
|a|b|c|
## 2. Pretrained model
Before running the training code, make sure you load the pre-trained weights. We provide pre-trained weights under [./models/pretrained](). You can also download Swin-Base model trained on ImageNet (through this [link]()).
## 3. Train
```
srun -p normal -w cluster -c 20 --mem=80G --gres=gpu:3080ti:4  -u sh train.sh 
```
## 4. Test
For model testing, we provide a python file to test our model by running python predict.sh.
```
srun -p normal -w cluster-2 -c 8 --mem=20G --gres=gpu:3080ti:1 -u sh predict.sh
```

## Citation
Please kindly consider citing our papers in your publications.
```
@inproceedings{Lu2026_supervise,
    title={Preserving Fairness Generalization in Deepfake Detection},
    author={Li Lin, Xinan He, Yan Ju, Xin Wang, Feng Ding, Shu Hu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024},
}
```
