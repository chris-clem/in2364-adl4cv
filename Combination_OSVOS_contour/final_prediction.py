import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_combo_img(contour_pred, OSVOS_img):
    contour_img = np.zeros(OSVOS_img.shape).astype(np.uint8)
    contour_pred = np.expand_dims(contour_pred.detach().numpy().astype(np.int32), axis=0)
    contour_img = cv2.fillPoly(contour_img, contour_pred, color=(255, 255, 255))
    contour_img = cv2.cvtColor(contour_img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    
    osvos_img = cv2.cvtColor(OSVOS_img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    osvos_img = np.where(osvos_img >= 255/2, 255, 0)

    combo_img = np.where(np.logical_and(osvos_img==255, contour_img==255), 1, 0)
    deletions_contour_img = np.where(np.logical_and(osvos_img!=255, contour_img==255), 1, 0)
    deletions_osvos_img = np.where(np.logical_and(osvos_img==255, contour_img!=255), 1, 0)
    
    return combo_img

def plot_all(contour_pred, OSVOS_img):
    contour_img = np.zeros(OSVOS_img.shape[0:2]).astype(np.uint8)
    contour_pred = np.expand_dims(contour_pred.detach().numpy().astype(np.int32), axis=0)
    contour_img = cv2.fillPoly(contour_img, contour_pred, color=255)

    osvos_img = cv2.cvtColor(OSVOS_img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    osvos_img = np.where(osvos_img >= 255/2, 255, 0)

    combo_img = np.where(np.logical_and(osvos_img==255, contour_img==255), 1, 0)
    deletions_contour_img = np.where(np.logical_and(osvos_img!=255, contour_img==255), 1, 0)
    deletions_osvos_img = np.where(np.logical_and(osvos_img==255, contour_img!=255), 1, 0)
    
    plt.title('CONTOUR IMAGE:')
    plt.imshow(contour_img)
    plt.show()

    plt.title('OSVOS IMAGE:')
    plt.imshow(osvos_img)
    plt.show()

    plt.title('COMBO IMAGE:')
    plt.imshow(combo_img)
    plt.show()

    plt.title('DELETIONS CONTOUR IMAGE:')
    plt.imshow(deletions_contour_img)
    plt.show()

    plt.title('DELETIONS OSVOS IMAGE:')
    plt.imshow(deletions_osvos_img)
    plt.show()
    
