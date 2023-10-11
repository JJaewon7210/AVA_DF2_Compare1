# YOWO with DeepFashion2 Training Project

This project focuses on training a model using the DeepFashion2 dataset while freezing the network of YOWO (You Only Watch Once).

![Project Image](TODO: Insert your project image path here)

## Features

One of the key features of this project is the ability to customize the loss function by using different 'build_target' methods. You can find these methods in the file 'utils/loss_ava.py'. Below are two options for updating the loss function:

**Option 1:** Update the 5 anchors (up, down, right, left, center) with the center as the center of the true label's bounding box.

```python
def df2_cls_loss(self, p, targets, BCEcls):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch zeros(1, device=device), torch.zeros(1, device=device)
    tcls,_tbox, indices, _anchors = self.build_targets_ver1(p, targets, cls_target=True)  # targets
```

**Option 2:** Update all anchors included in the true label.

python

```python
def df2_cls_loss(self, p, targets, BCEcls):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls,_tbox, indices, _anchors = this.build_targets_ver1(p, targets, cls_target=True)  # targets
```

Usage
-----

Follow these steps to use this project effectively:

1. **Clone the Repository:** Begin by cloning this repository to your local machine.

2. **Download Pretrained Weights:** Download the pretrained weights from the provided Google Drive links. Make sure to update the paths in the 'cfg/model.yaml' file.

3. **Download Datasets:** Download the required datasets, namely 'DeepFashion2' and 'AVA Activity Dataset.' Adjust the dataset paths in the 'cfg/ava.yaml' and 'cfg/deepfashion2.yaml' files.

4. **Training the Model:** Train the model using the 'train\_df2.py' script. You can monitor the training process using the WandB library. Customize hyperparameters and training options from the 'cfg/hyp.yaml' and 'cfg/deepfashion2.yaml' files.

5. **Evaluating the Model:** Evaluate the trained model by running the 'test\_df2.py' script.

Enjoy working with YOWO and DeepFashion2!

License
-------

This project is distributed under the MIT License.
