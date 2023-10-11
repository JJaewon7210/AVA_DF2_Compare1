# YOWO with DeepFashion2 Training Project

This project focuses on training a model using the DeepFashion2 dataset while freezing the network of YOWO (You Only Watch Once).

![image](https://github.com/JJaewon7210/AVA_DF2_Compare1/assets/96426723/6fde3595-0a69-45b5-99be-172d16405063)

![image](https://github.com/JJaewon7210/AVA_DF2_Compare1/assets/96426723/bab7f689-58a4-4956-8024-35a95f3a590b)


## Features  

One of the key features of this project is the ability to customize the loss function by using different 'build_target' methods. You can find these methods in the file 'utils/loss_ava.py'. Below are two options for updating the loss function:

**Option 1:**   
Update the 5 anchors (up, down, right, left, center) with the center as the center of the true label's bounding box.

```python
def df2_cls_loss(self, p, targets, BCEcls):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch zeros(1, device=device), torch.zeros(1, device=device)
    tcls,_tbox, indices, _anchors = self.build_targets(p, targets, cls_target=True)  # targets
```

**Option 2:**   
Update all anchors included in the true label.


```python
def df2_cls_loss(self, p, targets, BCEcls):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls,_tbox, indices, _anchors = this.build_targets_ver1(p, targets, cls_target=True)  # targets
```

Usage
-----

Follow these steps to use this project effectively:  

1. **Clone the Repository:**  
   Begin by cloning this repository to your local machine.

2.  **Download Pretrained Weights:**  
   You can obtain the pretrained weights from the following Google Drive links. Ensure to update the paths in the 'cfg/model.yaml' file.
    
    *   [resnext-101-kinetics.pth](https://drive.google.com/file/d/1633UbpB0UA73vuinYv19VZHNOY_825Vy/view?usp=sharing)
    *   [yolo.weights](https://drive.google.com/file/d/1lTNhAmaCm10W-uoCvdNsKSaEGoPBnHse/view?usp=sharing)
    *   [yowo_ava_16f_s1_best_ap_01790.pth](https://drive.google.com/file/d/1nk2Jkym3HCOP1ZIdZrvOgoZQYE8tivoB/view?usp=sharing)

3. **Download Datasets:**  
   Download the required datasets, namely 'DeepFashion2' and 'AVA 2.2 Activity Dataset.' Adjust the dataset paths in the 'cfg/ava.yaml' and 'cfg/deepfashion2.yaml' files.

4. **Training the Model:**  
   Train the model using the 'train_df2.py' script. You can monitor the training process using the WandB library. Customize hyperparameters and training options from the 'cfg/hyp.yaml' and 'cfg/deepfashion2.yaml' files.

5. **Evaluating the Model:**  
   Evaluate the trained model by running the 'test\_df2.py' script.

**Download Pretrained Weights:**   
You can obtain the pretrained weights. The following metrics are from the validation set of Deepfashion2.
We trained the model for 50 epochs. The model with 'augmentTrue' applied data augmentation, while 'augmentFalse' indicates training without data augmentation.  

   | Pretrained Weights | Download Link | Size (pixels) | mAP(50) | mAP(50-95) | Precision | Recall |
   | ------------------ | ------------- | ------------- | ------- | ---------- | --------- | ------ |
   | Option1_augmentTrue  | [Link 1](https://drive.google.com/file/d/1x7G0XZSX6z2DXAhlKO6Cxb8pbU7dtEBY/view?usp=sharing) | [224, 224] | 0.2392 | 0.2177 | 0.3352 | 0.3781 |
   | Option2_augmentFalse | [Link 2](https://drive.google.com/file/d/1DGrjDDPRc5AYTIhv8hMp5ws1IBfSv-Bl/view?usp=sharing) | [224, 224] | 0.2145 | 0.1918 | 0.4197 | 0.3118 |
   | Option2_augmentTrue  | [Link 3](https://drive.google.com/file/d/1QhEC4wugfKCVZSU7wvvCLLIfVywAWpSy/view?usp=sharing) | [224, 224] | 0.2399 | 0.2112 | 0.4281 | 0.3643 |

  
Enjoy working with YOWO and DeepFashion2!

License
-------

This project is distributed under the MIT License.
