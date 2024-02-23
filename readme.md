# YOWO with DeepFashion2 Training Project

This project focuses on training a model using the DeepFashion2 dataset while freezing the network of YOWO (You Only Watch Once).

![image](https://github.com/JJaewon7210/AVA_DF2_Compare1/assets/96426723/bab7f689-58a4-4956-8024-35a95f3a590b)


## Features  

One of the key features of this project is the ability to customize the loss function by using different 'build_target' methods. You can find these methods in the file `utils/loss_ava.py`.

**Option 1: Point**   
Update the single anchor. The single anchor is positioned at the center of the true label's bounding box. Since this method is used in YOLOv3, we named the function 'build_targets_for_YOLO3'.

```python
def df2_cls_loss(self, p, targets, BCEcls):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch zeros(1, device=device), torch.zeros(1, device=device)
    tcls, _tbox, indices, _anchors = self.build_targets_verYOLO3(p, targets)  # 1. Anchor assignment: point
```

**Option 2: Bidirectional method**   
Update all anchors included in the true label, which expands bidirectionally. For example, if the vertical_increase_ratio is set to 0.5, the height of the ground truth box will increase by 1 + 0.5 times its original height. Therefore, the box will expand upwards by 0.25 and downwards by 0.25. All anchors contained within it are then assigned.


```python
def df2_cls_loss(self, p, targets, BCEcls):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls,_tbox, indices, _anchors = this.build_targets_ver1(p, targets, cls_target=True, vertical_increase_ratio=0.5)  # targets
```

**Option 3: Unidirectional method**   
Update all anchors included in the true label, which expands in the unidirectional way. For example, if the vertical_increase_ratio is set to 0.5, the height of the ground truth box will increase by 1 + 0.5 times its original height. The direction of unidirectional expansion depends on the ground truth label. For instance, top items (i.e., short-sleeve top, long-sleeve top, outwear) will expand downwards, whereas bottom items (i.e., trousers, skirt) will expand upwards. One-piece items (i.e., short-sleeve dress, long-sleeve dress) remain unchanged in their expansion.


```python
def df2_cls_loss(self, p, targets, BCEcls):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, _tbox, indices, _anchors = self.build_targets_ver2(p, targets, cls_target=True, vertical_increase_ratio=0.5)  # 3. Anchor assignment: bidirectional

```

Usage
-----

Follow these steps to use this project effectively:  

1. **Clone the Repository:**  
   Begin by cloning this repository to your local machine.

2.  **Download Pretrained Weights:**  
   You can obtain the pretrained weights from the following Google Drive links. Ensure to update the paths in the `cfg/model.yaml` file.
    
    *   [resnext-101-kinetics.pth](https://drive.google.com/file/d/1633UbpB0UA73vuinYv19VZHNOY_825Vy/view?usp=sharing)
    *   [yolo.weights](https://drive.google.com/file/d/1lTNhAmaCm10W-uoCvdNsKSaEGoPBnHse/view?usp=sharing)
    *   [yowo_ava_16f_s1_best_ap_01790.pth](https://drive.google.com/file/d/1nk2Jkym3HCOP1ZIdZrvOgoZQYE8tivoB/view?usp=sharing)

3. **Download Datasets:**  
   Download the required datasets, namely `DeepFashion2` and `AVA 2.2 Activity Dataset`. Adjust the dataset paths in the `cfg/ava.yaml` and `cfg/deepfashion2.yaml` files.

4. **Training the Model:**  
   Train the model using the `train_df2.py` script. You can monitor the training process using the WandB library. Customize hyperparameters and training options from the `cfg/hyp.yaml` and `cfg/deepfashion2.yaml` files.

5. **Evaluating the Model:**  
   Evaluate the trained model by running the `test_df2.py` script.

**Download Pretrained Weights:**   
You can obtain the pretrained weights. The following metrics are from the validation set of Deepfashion2.
We trained the model for 100 epochs.

   | Pretrained Weights | Download Link | Size (pixels) | Precision | Recall | F1 Score | mAP(50) |
   | ------------------ | ------------- | ------------- | ------- | ---------- | --------- | ------ |
   | Point  | Comming soon.. | [224, 224] | 46.5 | 41.9 | 44.1 | 30.1 |
   | non-expansion  | Comming soon.. | [224, 224] | 59.0 | 49.8 | 50.5 | 39.0 |
   | bidirectional 50% | Comming soon.. | [224, 224] | 60.5 | 52.3 | 56.1 | 41.2 |
   | bidirectional 100%  | Comming soon.. | [224, 224] | 59.5 | 48.8 | 50.3 | 38.9 |
   | unidirectional 50% | Comming soon.. | [224, 224] | 60.8 | 55.0 | 56.7 | 43.3 |
   | unidirectional 100%  | Comming soon.. | [224, 224] | 60.7 | 57.7 | 58.6 | 45.0 |

  
Enjoy working with YOWO and DeepFashion2!

License
-------

This project is distributed under the MIT License.
