import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def phi_1(gradI, k):
    return np.exp(-np.square(gradI / k))


def phi_2(gradI, k):
    return 1 / (1 + np.square(gradI / k))


def Diffusion(img, n=30, k=50, g=phi_1, lamb=0.2):
    """
    Arguments:
        :img: ndarray to be filterred
        :n: number of time steps to compute
        :k: factor in conduction coefficient computation
        :method: [1|2] conduction function to use (phi_1 or phi_2)
        :lamb: (0<lamb<=0.25) lambda in the scheme (CFL)
    
    Returns:
        The image filterred
        
    According to the work of Pietro Perona and Jitendra Malik (November 1987)
    http://image.diku.dk/imagecanon/material/PeronaMalik1990.pdf
    """
    # If more than one channel, mean over them (convert from (nx,ny,3) to (nx,ny)
    if img.ndim == 3:
        img = img.mean(2)
    
    # Prepare arrays
    I = img.astype("float64").copy()
    z = np.zeros((4, *I.shape))
    
    # Iterate over "time"
    for i in range(n):
        D = z.copy()

        # Compute the gradients over the 4 directions
        D[0, :-1, :] = np.diff(I, axis=0)
        D[1, 1:, :] = -D[0, :-1, :]
        D[2, :, :-1] = np.diff(I, axis=1)
        D[3, :, 1:] = -D[2, :, :-1]

        # Multiply by diffusion coefficients
        gD = g(np.absolute(D), k) * D
        # Apply scheme
        I += lamb * np.sum(gD, axis=0)
        
    return I


def filter_mid(img, eps=0.01, mid=0.5):
    edges = np.where(np.logical_and(img>mid-eps, img<mid+eps))
    img = np.zeros_like(img)
    img[edges] = 1
    return img


def load_itk(filename):
    return sitk.GetArrayFromImage(sitk.ReadImage(filename))

        
def plot(*data, figsize=None):
    """
    Arguments:
        :data: list of dict. each dict represents an image to plot with keys: 
            - `image`: numpy.ndarray,
            - `title`: str,
            - `cmap`: str,
            - `colorbar`: bool
            Only image is required.
    """
    fig, axs = plt.subplots(1, len(data), figsize=figsize)

    for ax, d in zip(axs, data):
        c = ax.imshow(d.get("image"), cmap=d.get("cmap", "Greys"))
        if d.get("title") is not None:
            ax.set_title(d.get("title"))
        if d.get("colorbar", False):
            fig.colorbar(c)

    fig.canvas.draw()