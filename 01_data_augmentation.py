"""First step of our pipeline: augment DAVIS images.

Augmentations cannot happen on the fly as for each image, the contour needs to be extracted,
the translations computed, and then for each contour point the OSVOS feature vectors extracted.
"""

import os
import random

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import src.config as cfg

# OSVOS like augmentations
class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample, random):

        if random < 0.5:
            for elem in sample.keys():
                if 'fname' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample
    
class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth."""
    
    def __call__(self, sample, rot, sc):

        for elem in sample.keys():
            if 'fname' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample


def save_sample(sample, frame, augmentation_count,
                annotations_augmented_folders_path, 
                images_augmented_folders_path):
    """Save sample (image annotation pair).
    
    Parameters
    ----------
    sample : dict
        Dict containing image and annotation
    frame : str
        Name of the sample
    augmentation_count : int
        Index of the augmentation (0 for original sample)
    annotations_augmented_folders_path : str
        Path to where augmented annotation should be stored
    images_augmented_folders_path : str
        Path to where augmented images should be stored
    """
    
    # Save image
    file_name_img = '{}.jpg'.format(frame[:5])
    image = sample['image'].astype(np.uint8)
    #image = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
    image_save_path = os.path.join(images_augmented_folders_path, 
                                   augmentation_count)
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    imageio.imwrite(os.path.join(image_save_path, file_name_img), image)
    
    # Save annotation
    file_name_annot = '{}.png'.format(frame[:5])
    annotation = sample ['gt']
    annotation = (255.0 / annotation.max() * (annotation - annotation.min())).astype(np.uint8)
    annotation_save_path = os.path.join(annotations_augmented_folders_path, 
                                        augmentation_count)
    if not os.path.exists(annotation_save_path):
        os.makedirs(annotation_save_path)
    imageio.imwrite(os.path.join(annotation_save_path, file_name_annot), annotation)

    
def augment_data(annotations_folders_path, images_folders_path,
                 annotations_augmented_folders_path, images_augmented_folders_path,
                 meanval, augmentation_count):
    """Augment DAVIS data (augmentations and images) using rotations and scales.
    
    Parameters
    ----------
    annotations_folders_path : str
        Path to DAVIS augmentations
    images_folders_path : str
        Path to DAVIS images
    annotations_augmented_folders_path : str
        Path to where augmented annotation should be stored
    images_augmented_folders_path : str
        Path to where augmented images should be stored
    meanval : tuple
        Mean value for image normalization
    augmentation_count : int
        Number of augmentations to create
    """

    # Create augmentation_count augmentations
    randoms = []
    random_rots = []
    random_scales = []
    rots=(-30, 30)
    scales=(.75, 1.25)
    
    for i in range(augmentation_count):
        randoms.append(random.random())
        
        rot = (rots[1] - rots[0]) * random.random() - \
              (rots[1] - rots[0])/2
        random_rots.append(rot)
        
        sc = (scales[1] - scales[0]) * random.random() - \
             (scales[1] - scales[0]) / 2 + 1
        random_scales.append(sc)
        
    random_horizontal_flip = RandomHorizontalFlip()
    scale_and_rotate = ScaleNRotate()
    
    # Get list of sequences
    sequences = os.listdir(images_folders_path)
    sequences.sort()
    
    # Iterate through sequences
    for i, sequence in enumerate(sequences):

        # Debugging
        if (i > cfg.DEBUG): break

        print('#{}: {}'.format(i, sequence))
        
        # Create folders to save augmented annotations and images
        annotations_aug_folder_path = os.path.join(annotations_augmented_folders_path, sequence)
        if not os.path.exists(annotations_aug_folder_path):
            os.makedirs(annotations_aug_folder_path)
            
        images_aug_folder_path = os.path.join(images_augmented_folders_path, sequence)
        if not os.path.exists(images_aug_folder_path):
            os.makedirs(images_aug_folder_path)

        # Get list of frames
        frames = os.listdir(os.path.join(images_folders_path, sequence))
        if '.ipynb_checkpoints' in frames:
            frames.remove('.ipynb_checkpoints')
        frames.sort()
        
        augmentation_blacklist = []
        
        # Iterate through frames
        for j, frame in enumerate(frames):

            # Debugging
            if (j > cfg.DEBUG): break
            #print('\t#{}: {}'.format(j, frame))
            
            # Skip these sequences as annotations are completely black
            if (sequence == 'bmx-bumps' and frame == '00059.jpg'): break
            if (sequence == 'surf' and frame == '00053.jpg'): break
                
            # Load annotation and image
            annotation_path = os.path.join(annotations_folders_path, sequence, frame[:5] + '.png')
            image_path = os.path.join(images_folders_path, sequence, frame)
            
            annotation = cv2.imread(annotation_path)
            annotation = np.array(annotation, dtype=np.float32)
            annotation = annotation/np.max([annotation.max(), 1e-8])
            
            image = imageio.imread(image_path)
            image = np.array(image, dtype=np.float32)
            image = np.subtract(image, np.array(meanval, dtype=np.float32))
                     
            # Create sample
            sample = {'image': image, 'gt': annotation}
            
            # Save original sample
            save_sample(sample, frame, '0',
                        annotations_aug_folder_path, images_aug_folder_path)
            
            # If val sequence, do not augment
            if sequence not in cfg.TRAIN_SEQUENCES: continue
            
            # Apply augmentations and save them
            for k in range(augmentation_count):
                if k not in augmentation_blacklist:
                    #print('\t\tAugmentation #{}'.format(k+1))
                    sample = random_horizontal_flip(sample, randoms[k])
                    sample = scale_and_rotate(sample, random_rots[k], random_scales[k])

                    # If annotation is completely black, don't save it
                    if (np.sum(sample['gt']) == 0): 
                        print('\t\t{} Augmentation #{}: Annotation black'.format(frame, k+1))
                        augmentation_blacklist.append(k)
                        continue
                
                    save_sample(sample, frame, str(k+1),
                                annotations_aug_folder_path, images_aug_folder_path)


if __name__ == "__main__":                    
    augment_data(cfg.ANNOTATIONS_FOLDERS_PATH, cfg.IMAGES_FOLDERS_PATH,
                 cfg.ANNOTATIONS_AUGMENTED_FOLDERS_PATH, cfg.IMAGES_AUGMENTED_FOLDERS_PATH,
                 cfg.MEANVAL, cfg.AUGMENTATION_COUNT)