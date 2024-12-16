import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def imshow(img, save_path=None):
    """
    Display and optionally save a tensor image.
    Args:
        img (Tensor): The image to display (C, H, W).
        save_path (str): Path to save the image. If None, image won't be saved.
    """
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    plt.imshow(npimg)
    plt.axis('off')  # Remove axes for a cleaner display
    plt.show()

    if save_path:
        plt.imsave(save_path, npimg)

def display_images(data_loader, classes, num_images=8, save_path=None):
    """
    Display a grid of images from a DataLoader.
    Args:
        data_loader (DataLoader): PyTorch DataLoader to fetch images from.
        classes (list): List of class names.
        num_images (int): Number of images to display.
        save_path (str): Path to save the grid image. If None, grid won't be saved.
    """
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # Select only the required number of images
    images = images[:num_images]
    labels = labels[:num_images]

    # Make a grid of images
    img_grid = vutils.make_grid(images, nrow=num_images)

    # Display the grid
    imshow(img_grid, save_path=save_path)

    # Print class labels
    print(' '.join(f'{classes[label]:5s}' for label in labels))
