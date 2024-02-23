import os
import torch
import cv2
import numpy as np
from utils.general import non_max_suppression, xyxy2xywh
from utils.plots import read_labelmap, plot_batch_image_from_preds

NUM_FRAMES = 16
STRIDE = 8
IMG_SIZE = 224

def load_images_to_tensor(image_paths, img_size=224):
    frame_sequence = []
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        h, w, _ = frame.shape

        # Crop the image to a square by removing the excess part from the longer side
        if h > w:
            start_y = (h - w) // 2
            frame_cropped = frame[start_y:start_y + w, :]
        else:
            start_x = (w - h) // 2
            frame_cropped = frame[:, start_x:start_x + h]

        # Resize the cropped square image to the target size
        frame_resized = cv2.resize(frame_cropped, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # Convert the resized frame to CHW format and normalize
        img = frame_resized.transpose([2, 0, 1])  # Convert to CHW format
        img = img / 255.0  # Normalize to [0, 1]
        img = np.ascontiguousarray(img).astype(np.float32)

        frame_sequence.append(img)

    imgs = np.stack(frame_sequence, axis=1)  # Shape: (C, T, H, W)
    imgs = torch.from_numpy(imgs)
    imgs = torch.unsqueeze(imgs, 0)  # Add an extra batch dimension
    # return imgs, np.expand_dims(frame_cropped, axis=0)  # Return the last resized frame for reference
    return imgs, np.expand_dims(frame_resized, axis=0)  # Return the last resized frame for reference



def process_image_sequence(weight_path, image_folder, save_path, save_txt_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load(weight_path, map_location=device)
    if type(model) == dict:
        model = model.get('ema', model['model'])
    model.eval()

    labelmap_ava, _ = read_labelmap("D:/Data/AVA/annotations/ava_action_list_v2.2.pbtxt")
    labelmap_df2, _ = read_labelmap("D:/Data/DeepFashion2/df2_list.pbtxt")


    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')])
    
    for i in range(0, len(image_paths) - NUM_FRAMES + 1, STRIDE):
        current_batch = image_paths[i:i + NUM_FRAMES]
        imgs, keyframe = load_images_to_tensor(current_batch, IMG_SIZE)
        _, _, _, height, width = imgs.shape
        imgs = imgs.to(device, non_blocking=True).half()

        with torch.no_grad():
            out_bboxs, out_clos, out_acts = model(imgs)
            preds_clo = torch.cat((out_bboxs[0], out_clos[0]), dim=2)
            preds_act = torch.cat((out_bboxs[0], out_acts[0]), dim=2)

        save_path_clo = os.path.join(save_path, f'clo_results_{str(i // STRIDE).zfill(5)}.jpg')
        save_path_act = os.path.join(save_path, f'act_results_{str(i // STRIDE).zfill(5)}.jpg')
        save_path_clo_txt = os.path.join(save_txt_path, f'clo_results_{str(i // STRIDE).zfill(5)}.txt')
        save_path_act_txt = os.path.join(save_txt_path, f'act_results_{str(i // STRIDE).zfill(5)}.txt')

        outs_clo = non_max_suppression(preds_clo, conf_thres=0.5, iou_thres=0.5, cls_thres=0.3, multi_label=True)
        plot_batch_image_from_preds(keyframe.copy(), outs_clo, save_path_clo, labelmap_df2, (width*2, height*2) )
        if save_txt_path:
            for *xyxy, conf, cls in outs_clo[0].tolist():
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # already normalized xywh
                line = (cls, *xywh, conf)
                with open(save_path_clo_txt, 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
        
        outs_act = non_max_suppression(preds_act, conf_thres=0.5, iou_thres=0.5, cls_thres=0.3, multi_label=False)
        plot_batch_image_from_preds(keyframe.copy(), outs_act, save_path_act, labelmap_ava, (width*2, height*2) )
        if save_txt_path:
            for *xyxy, conf, cls in outs_act[0].tolist():
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # already normalized xywh
                line = (cls, *xywh, conf)
                with open(save_path_act_txt, 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

def process_videos_folder(images_root, infers_root, weight_path, save_txt_path):
    print('Start!')
    for root, dirs, files in os.walk(images_root):
        total = len(dirs)  # Total number of directories to process
        if total > 0:
            print(f"Found {total} directories in {root}. Starting processing...")
        else:
            print(f"No directories found in {root}. Skipping...")
            continue  # Skip to the next iteration of os.walk

        for n, dir in enumerate(dirs, start=1):  # Start counting from 1
            image_folder = os.path.join(root, dir)
            print(f"Processing {n}/{total}: {image_folder}")

            # Format the directory index as a zero-padded string (e.g., 001, 002)
            dir_index = f"{n:03d}"

            # Modify the save path to include the numbering and "#" in front of the folder name
            relative_path = os.path.relpath(image_folder, images_root)
            modified_relative_path = dir_index + '_' + os.path.join(*relative_path.split(os.path.sep))
            save_path = os.path.join(infers_root, modified_relative_path)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
                process_image_sequence(weight_path, image_folder, save_path, save_txt_path)
                print(f"Finished processing {n}/{total} and saved results to: {save_path}")
            else:
                print(f"Directory already exists: {save_path}. Skipping...")

            print(f"Completed {n}/{total}: {image_folder}\n")  # Extra newline for readability

        if total > 0:
            print(f"Completed all {total} directories in {root}.\n")

def process_video(image_folder, images_root, infers_root, weight_path, save_txt_path):
    print(f"Processing 1/1: {image_folder}")

    # Format the directory index as a zero-padded string (e.g., 001, 002)
    n = 0

    # Modify the save path to include the numbering and "#" in front of the folder name
    relative_path = os.path.relpath(image_folder, images_root)
    modified_relative_path = os.path.join(*relative_path.split(os.path.sep))
    save_path = os.path.join(infers_root, modified_relative_path)
    save_txt_path = os.path.join(save_txt_path, modified_relative_path)
    
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        process_image_sequence(weight_path, image_folder, save_path, save_txt_path)
        print(f"Finished processing 1/1 and saved results to: {save_path}")
    else:
        print(f"Directory already exists: {save_path}. Skipping...")

    print(f"Completed 1/1: {image_folder}\n")  # Extra newline for readability
    

if __name__ == '__main__':

    images_root = 'D:/Data/tapo_camera/images'
    infers_root = 'D:/Data/tapo_camera/infers'
    weight_path = 'runs/train/build_target_ver2_vertical100_augO_epoch100/weights/epoch99.pt'
    save_txt_path = 'D:/Data/tapo_camera/preds'
    image_folder = images_root + '/' + 'detected_20240129_232335'
    process_video(image_folder, images_root, infers_root, weight_path, save_txt_path)
