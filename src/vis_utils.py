import cv2
import matplotlib.pyplot as plt
import numpy as np


def close_image(image, closing_kernel_size):
    '''Returns the image that is closed with a elliptical kernel.'''
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(closing_kernel_size, closing_kernel_size))
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    return closing


def compute_combo_img(contour_pred, osvos_img):
    '''Create combo image from predicted contour and osvos prediction.'''
    
    contour_pred = np.expand_dims(contour_pred.detach().numpy().astype(np.int32), axis=0)
    contour_img = np.zeros_like(osvos_img, dtype=np.uint8)
    contour_img = cv2.fillPoly(contour_img, contour_pred, color=(255, 255, 255))
    contour_img = cv2.cvtColor(contour_img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    
    # Dilate our image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    contour_img = cv2.dilate(contour_img, kernel, iterations=1)
    
    osvos_img = cv2.cvtColor(osvos_img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    osvos_img = np.where(osvos_img >= 255/2, 255, 0)

    combo_img = np.where(np.logical_and(osvos_img==255, contour_img==255), 1, 0)
    deletions_contour_img = np.where(np.logical_and(osvos_img!=255, contour_img==255), 1, 0)
    deletions_osvos_img = np.where(np.logical_and(osvos_img==255, contour_img!=255), 1, 0)
    
    return contour_img, combo_img, deletions_contour_img, deletions_osvos_img


def extract_longest_contour(image, closing_kernel_size, method):
    '''Returns the contour with the most points from a given image.'''
    
    # Close image
    image_closed = close_image(image, closing_kernel_size)
    
    # Apply threshold to turn it into binary image
    ret, thresh = cv2.threshold(image_closed, 127, 255, 0)

    # Find contour
    # Change method for different number of points:
    # CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS
    contours, _ = cv2.findContours(image=thresh,
                                   mode=cv2.RETR_TREE,
                                   method=method)
    
    # Get longest contour from contours
    longest_contour = max(contours, key=len).astype(np.float32)
    
    return longest_contour


def load_gray_img(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def plot_combo_img(contour_pred, osvos_img):
    (contour_img, combo_img, 
     deletions_contour_img, deletions_osvos_img) = compute_combo_img(contour_pred, osvos_img)
    
    fig = plt.figure(figsize=(10,25))

    ax = plt.subplot(511)
    ax.set_title('Contour Image')
    ax.imshow(contour_img, cmap='gray')
    
    ax = plt.subplot(512)
    ax.set_title('OSVOS Image')
    ax.imshow(osvos_img, cmap='gray')

    ax = plt.subplot(513)
    ax.set_title('Combined Image')
    ax.imshow(combo_img, cmap='gray')

    ax = plt.subplot(514)
    ax.set_title('Deletions Contour Image')
    ax.imshow(deletions_contour_img, cmap='gray')

    ax = plt.subplot(515)
    ax.set_title('Deletions OSVOS Image')
    ax.imshow(deletions_osvos_img, cmap='gray')
    
    plt.show()
    

def plot_img_with_contour_and_translation(img, contour, translation_gt):
    
    # Plot image
    plt.imshow(img)
    
    # Plot contour
    contour = contour.numpy()
    plt.scatter(contour[:, 0], contour[:, 1], s=80, marker='.', c='b')

    # Plot ground truth translation
    translation_gt = translation_gt.numpy()
    for c, t in zip(contour, translation_gt):
        plt.arrow(c[0], c[1],
                  t[0], t[1],
                  width=1, color='g')
    
    plt.show()
    

def plot_loss(solver):
    
    plt.figure(figsize=(10,7))
    plt.plot(solver.loss_epoch_history['translation_loss_L2'], '-', label='train loss')
    plt.plot(solver.val_loss_history, '-', label='val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    

def plot_translations(img, contour, translation_gt, translation_pred):
    
    # Plot image
    plt.imshow(img)    
    
    # Plot ground truth next contour
    contour = contour.numpy()
    translation_gt = translation_gt.numpy()
    next_contour_gt = contour + translation_gt
    plt.scatter(next_contour_gt[:, 0], next_contour_gt[:, 1], s=80, marker='.', c='g')
    
    # Plot predicted next contour
    translation_pred = translation_pred.numpy()
    next_contour_pred = contour + translation_pred
    plt.scatter(next_contour_pred[:, 0], next_contour_pred[:, 1], s=80, marker='.', c='r')
    
    plt.show()

