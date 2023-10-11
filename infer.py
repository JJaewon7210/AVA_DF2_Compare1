
import os
from threading import Thread
import torch
from torchvision import transforms
import cv2
import yaml
import numpy as np
from utils.general import non_max_suppression, ConfigObject
from utils.plots import read_labelmap, un_normalized_images, plot_batch_image_from_preds
from model.YOWO import YOWO_CUSTOM as Model

def load_images_to_tensor(image_folder='inference/example_video', num_images=16, channel=3, img_size=224):
    # List all image file names in the folder
    frame_sequence = []
    image_paths = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    image_paths = image_paths[:num_images]
    for image_path in image_paths:
        frame = cv2.imread(image_folder+'/'+image_path)
        frame_sequence.append(frame)
    
    # Resize and process the image frames in the sequence
    imgs = [cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR) for frame in frame_sequence]

    # Convert image frames to CHW format (assuming cv2_transform.HWC2CHW performs the required conversion)
    imgs = [img.transpose([2,0,1]) for img in imgs]
    
    # Image [0, 255] -> [0, 1].
    imgs = [img / 255.0 for img in imgs]

    imgs = [
        np.ascontiguousarray(
            img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
        ).astype(np.float32)
        for img in imgs
    ]
    
    # Concat list of images to single ndarray.
    imgs = np.concatenate(
        [np.expand_dims(img, axis=1) for img in imgs], axis=1
    )

    imgs = np.ascontiguousarray(imgs)
    imgs = torch.from_numpy(imgs)
    imgs = torch.unsqueeze(imgs, 0)
    return imgs, np.expand_dims(frame, axis=0)

def main(weight_path, image_folder='inference/example_video'):
    # Device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    
    # Load model
    model = torch.load(weight_path, map_location=device)
    if type(model) == dict:
        if 'ema' in model:
            model = model['ema']
        else:
            model = model['model']

    # Configure
    model.half()
    model.eval()
    labelmap_ava, _ = read_labelmap("D:/Data/AVA/annotations/ava_action_list_v2.2.pbtxt")
    labelmap_df2, _ = read_labelmap("D:/Data/DeepFashion2/df2_list.pbtxt")

    # load images
    imgs, keyframe = load_images_to_tensor(image_folder)
    _, height, width, _ = keyframe.shape
    imgs = imgs.to(device, non_blocking=True)
    imgs = imgs.half() # uint8 to fp16/32

    with torch.no_grad():
        # Run model
        out_bboxs, out_clos, out_acts = model(imgs)
        
        out_bbox_infer, out_bbox_features = out_bboxs[0], out_bboxs[1]
        out_clo_infer, out_clo_features = out_clos[0], out_clos[1]
        out_act_infer, out_act_features = out_acts[0], out_acts[1]
        preds_clo = torch.cat((out_bbox_infer, out_clo_infer), dim=2)
        preds_act = torch.cat((out_bbox_infer, out_act_infer), dim=2)
        
    # Plot images

    save_path_clo = 'inference/infer_results(clo).jpg'
    save_path_act = 'inference/infer_results(act).jpg'
    
    outs_clo = non_max_suppression(preds_clo, conf_thres=0.5, iou_thres=0.5, cls_thres=0.25, multi_label=True)
    plot_batch_image_from_preds(keyframe.copy(), outs_clo, save_path_clo, labelmap_df2, (width, height)*2 )
    
    outs_act = non_max_suppression(preds_act, conf_thres=0.5, iou_thres=0.5)
    plot_batch_image_from_preds(keyframe.copy(), outs_act, save_path_act, labelmap_ava, (width, height)*2 )

if __name__ == '__main__':
    
    main(weight_path='runs/train/AVA_DF22/weights/last.pt')