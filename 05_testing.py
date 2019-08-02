"""Fith step of our approach: test our predictions using val sequences."""

import os

import cv2
import numpy as np
import pandas as pd
import torch

from pg_networks.gcn import GCN
from pg_networks.dynamic_edge import DynamicEdge
import src.config as cfg
from src.create_data import create_osvos_model, create_data
from src.metrics import db_eval_iou, db_eval_boundary
from src.vis_utils import compute_combo_img, extract_longest_contour, load_gray_img


def testing(use_parent_model_results, simple_contour_prediction=False, save_results=False):
    mean_Js_combo = []
    mean_Js_osvos = []

    mean_Fs_combo = []
    mean_Fs_osvos = []
    
    metrics = {}

    # Iterate through val sequences
    for i, sequence in enumerate(cfg.VAL_SEQUENCES):

        # Debugging
        #if i > 1: break
        print('#{}: {}'.format(i, sequence))

        # Get path to images and annotations
        raw_images_path = os.path.join('pg_datasets/DAVIS_2016/raw/Images',
                                       sequence, '0')
        raw_annotations_path = os.path.join('pg_datasets/DAVIS_2016/raw/Annotations', 
                                            sequence, '0')

        # Get list of frames
        frames = os.listdir(raw_images_path)
        if '.ipynb_checkpoints' in frames:
            frames.remove('.ipynb_checkpoints')
        frames.sort()

        Js_combo = []
        Js_osvos = []

        Fs_combo = []
        Fs_osvos = []    

        # Iterate through frames
        for j, frame in enumerate(frames[:-1]):

            # Debugging
            #if j > 3: break
            
            # If first frame, extract contour from gt annotation
            if j == 0:
                annotation_0_path = os.path.join(raw_annotations_path, 
                                                 frame[:5] + '.png')
                annotation_0_gray = load_gray_img(annotation_0_path)
                
                contour_0 = extract_longest_contour(annotation_0_gray, 
                                                    cfg.CLOSING_KERNEL_SIZE, 
                                                    cv2.CHAIN_APPROX_TC89_KCOS)
                contour_0 = np.squeeze(contour_0)

            # Create data object
            image_path_0 = os.path.join(raw_images_path, frames[j])
            image_path_1 = os.path.join(raw_images_path, frames[j+1])
            data = create_data(contour_0, None, image_path_0, image_path_1, 
                               osvos_model, cfg.K)

            # Forward pass to get outputs
            with torch.no_grad():
                translation_0_1_pred = model(data)    

            # Compute contour_1 = contour_0 + translation_0_1_pred
            contour_1_pred = np.add(contour_0, translation_0_1_pred)

            # Load OSVOS result image
            if use_parent_model_results:
                osvos_results = cfg.PARENT_MODEL_RESULTS_FOLDERS_PATH
            else: 
                osvos_results = cfg.OSVOS_RESULTS_FOLDERS_PATH
                
            osvos_img_1_path = os.path.join(osvos_results, sequence, 
                                            frames[j+1][:5] + '.png')
            osvos_img_1 = cv2.imread(osvos_img_1_path)
            osvos_img_1_gray = cv2.imread(osvos_img_1_path, cv2.IMREAD_GRAYSCALE)

            # Create combined image
            _, combo_img_1, _, _ = compute_combo_img(contour_1_pred, osvos_img_1)

            # Save results
            if save_results:
                if not os.path.exists(os.path.join(cfg.COMBO_RESULTS_FOLDERS_PATH, 
                                                   sequence)):
                    os.makedirs(os.path.join(cfg.COMBO_RESULTS_FOLDERS_PATH, 
                                             sequence))
                    
                combo_img_1_path = os.path.join(cfg.COMBO_RESULTS_FOLDERS_PATH, 
                                                sequence, frames[j+1][:5] + '.png')
                cv2.imwrite(combo_img_1_path, combo_img_1*255)

                combo_img_1_blended_path = os.path.join(cfg.COMBO_RESULTS_FOLDERS_PATH, 
                                           sequence, frames[j+1][:5] + '_blended.png') 
                blended = (0.4 * cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE) + 
                           (0.6 * combo_img_1*255)).astype("uint8")
                cv2.imwrite(combo_img_1_blended_path, blended)
            
            # Load ground truth annotation
            annotation_1_path = os.path.join(raw_annotations_path, 
                                             frames[j+1][:5] + '.png')
            annotation_1_gray = cv2.imread(annotation_1_path, cv2.IMREAD_GRAYSCALE)

            #Compute J
            J_combo = db_eval_iou(annotation_1_gray, combo_img_1)
            J_osvos = db_eval_iou(annotation_1_gray, osvos_img_1_gray)

            #Compute F
            F_combo = db_eval_boundary(combo_img_1, annotation_1_gray)
            F_osvos = db_eval_boundary(osvos_img_1_gray, annotation_1_gray)

            Js_combo.append(J_combo)
            Js_osvos.append(J_osvos)
            Fs_combo.append(F_combo)
            Fs_osvos.append(F_osvos)
            
            #if combo image completely dark we lost object and cannot recover
            if np.sum(combo_img_1) == 0:
                print('{} {} Combo image completely dark'.format(sequence, frame[:5]))
                break
            
            if simple_contour_prediction == True:
                contour_0 = contour_1_pred.numpy()
            else:
                contour_0 = extract_longest_contour(np.uint8(combo_img_1*255), 
                                                    cfg.CLOSING_KERNEL_SIZE, 
                                                    cv2.CHAIN_APPROX_TC89_KCOS)
                contour_0 = np.squeeze(contour_0)

        mean_J_combo = np.mean(np.array(Js_combo))
        mean_J_osvos = np.mean(np.array(Js_osvos))
        mean_F_combo = np.mean(np.array(Fs_combo))
        mean_F_osvos = np.mean(np.array(Fs_osvos))
        
        metrics[sequence] = [mean_J_combo, mean_J_osvos, mean_F_combo, mean_F_osvos]

        mean_Js_combo.append(mean_J_combo)
        mean_Js_osvos.append(mean_J_osvos)
        mean_Fs_combo.append(mean_F_combo)
        mean_Fs_osvos.append(mean_F_osvos)

    columns = ['mean_J_combo', 'mean_J_osvos', 'mean_F_combo', 'mean_F_osvos']
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=columns)
    
    metrics_df['mean_J_difference'] = metrics_df.apply(
        lambda row: row.mean_J_combo - row.mean_J_osvos, axis=1)
    metrics_df['mean_F_difference'] = metrics_df.apply(
        lambda row: row.mean_F_combo - row.mean_F_osvos, axis=1)
    
    metrics_df = metrics_df.append(metrics_df.describe())
    print(metrics_df)
    
    if use_parent_model_results:
        filename = cfg.BEST_MODEL[:-4] + '_parent.csv'
    else: 
        filename = cfg.BEST_MODEL[:-4] + '_osvos.csv'
        
    metrics_df.to_csv(filename)

if __name__ == "__main__": 
    # Load OSVOS to extract feature vectors
    osvos_model = create_osvos_model(cfg.PARENT_MODEL_PATH, cfg.LAYER)
    
    # Load GCN model to get predictions
    model_path = cfg.BEST_MODEL
    model = GCN(512, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.double()
    
    # Start testing
    testing(cfg.USE_PARENT_MODEL_RESULTS)