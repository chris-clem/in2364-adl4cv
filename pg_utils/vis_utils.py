import matplotlib.pyplot as plt
import numpy as np

def show_img_with_contour(data):
    img = data.img.numpy().astype(np.int64)
    img = np.squeeze(img)
    img = np.moveaxis(img, 0, 2)
    
    contour = data.contour.numpy()
    
    plt.imshow(img)
    plt.scatter(contour[:, 0], contour[:, 1])
    
    plt.show()
    
def plot_translation(image, translation_ground_truth, translation_pred):
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    ax.scatter(translation_ground_truth[:, 0], translation_ground_truth[:, 1], color='g')
    ax.scatter(translation_pred[:, 0], translation_pred[:, 1], color='r')
    
    # Plot image
    ax.imshow(image)
    
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    
def show_loss(solver):
    plt.plot(solver.train_loss_epoch_history, '-', label='train loss')
    plt.plot(solver.val_loss_history, '-', label='val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
