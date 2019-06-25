import matplotlib.pyplot as plt
import numpy as np


def plot_img_with_contour_and_translation(img, contour, translation_gt):
    
    # Plot image
    img = img.numpy().astype(np.int64)
    img = np.squeeze(img)
    img = np.moveaxis(img, 0, 2)
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
    

def plot_translations(img, contour, translation_gt, translation_pred):
    
    # Plot image
    img = img.numpy().astype(np.int64)
    img = np.squeeze(img)
    img = np.moveaxis(img, 0, 2)
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
    
    
def plot_loss(solver):
    
    plt.figure(figsize=(10,7))
    plt.plot(solver.train_loss_epoch_history, '-', label='train loss')
    plt.plot(solver.val_loss_history, '-', label='val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
