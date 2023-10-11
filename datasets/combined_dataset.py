from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):

        self.datasetDF2 = dataset1
        self.datasetAVA = dataset2
        self.length = max(len(dataset1), len(dataset2))
        self.index2 = 0
        
        self.print_summary()
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index1 = index % len(self.datasetDF2)
        index2 = self.index2 % len(self.datasetAVA)

        item1 = self.datasetDF2[index1]
        item2 = self.datasetAVA[index2]
        
        self.index2 += 1  # Increment index2 after accessing the item
        
        return item1, item2
    
    def print_summary(self):
        ratio = len(self.datasetAVA) / len(self.datasetDF2)
        logger.info(f"\n====> Loading Combined dataset")
        logger.info(f"Name of dataset1: {type(self.datasetDF2).__name__}")
        logger.info(f"Name of dataset2: {type(self.datasetAVA).__name__}")
        logger.info(f"Length of dataset:  {self.length}, Defined by max length of dataset1 ({len(self.datasetDF2)}) and dataset2 ({len(self.datasetAVA)})")
        logger.info(f"For every appearance of dataset1, dataset2 appears approximately {ratio:.2f} times.")

    @staticmethod
    def collate_fn(batch):
        '''
        Collate function to preprocess and organize a batch of data for a deep learning model.

        Arguments:
            batch (list): A list of tuples, where each tuple contains two items: item1 and item2.

        Returns:
            tuple: A tuple containing two elements - item1_batchs and item2_batchs.
                item1_batchs contains preprocessed data from item1.
                item2_batchs contains preprocessed data from item2.
        '''

        # Unpack the batch into two separate lists: item1_batch and item2_batch
        item1_batch, item2_batch = zip(*batch)

        # Process item1_batch data
        # Extract individual components from each item1 tuple
        imgs, labels, paths, _shapes, features = zip(*item1_batch)
        # Explanation of variables in item1_batch:
        #   - imgs (torch.uint8): Image data with shape [B, 3, H, W]
        #   - labels (torch.float32): Label data with shape [num, 6], where num is the number of objects in the image.
        #                            Each row in the labels contains: [Batch_num, class_num, x, y, h, w].
        #   - paths (tuple[str]): List of image paths with length B.
        #   - _shapes (tuple[tuple]): List of tuples, each containing (h0, w0), ((h / h0, w / w0), pad).
        #   - features (torch.float32): Features data with shape [B, 15, 7, 7], representing 3 (num_anchor) x 5 (bbox xywh + confidence score).

        # Modify the label data to include the target image index for build_targets()
        for i, l in enumerate(labels):
            l[:, 0] = i
        # Explanation: Loop through each label in the labels list and set the first column to the image index (i) to use in build_targets().
        
        # Stack the processed item1_batch components into tensors
        item1_batchs = (torch.stack(imgs, dim=0), torch.cat(labels, dim=0), paths, _shapes, torch.stack(features, dim=0))
        # Explanation: Stack individual components into tensors and create the final item1_batchs tuple.

        # Process item2_batch data
        # Extract individual components from each item2 dictionary
        clips = torch.stack([item["clip"] for item in item2_batch])
        cls = np.stack([item["cls"] for item in item2_batch])
        boxes = np.stack([item["boxes"] for item in item2_batch])
        feature_s = np.stack([item["feature_s"] for item in item2_batch])
        feature_m = np.stack([item["feature_m"] for item in item2_batch])
        feature_l = np.stack([item["feature_l"] for item in item2_batch])
        # Explanation of variables in item2_batch:
        #   - clips (torch.float32): Video clips data with shape [B, 3, T, H, W].
        #   - cls (np.array, dtype('float32')): Array of class data with shape [B, 50, 80].
        #                                      It contains up to 50 labels (from the beginning to the end).
        #   - boxes (np.array, dtype('float32')): Array of bounding box data with shape [B, 50, 4].
        #   - feature_s (np.array, dtype('float16')): Array of small feature data with shape [B, 3, 7, 7, 18].
        #   - feature_m (np.array, dtype('float16')): Array of medium feature data with shape [B, 3, 14, 14, 18].
        #   - feature_l (np.array, dtype('float16')): Array of large feature data with shape [B, 3, 28, 28, 18].

        # Stack the processed item2_batch components into tensors
        item2_batchs = (clips, cls, boxes, feature_s, feature_m, feature_l)
        # Explanation: Stack individual components into tensors and create the final item2_batchs tuple.

        return item1_batchs, item2_batchs

