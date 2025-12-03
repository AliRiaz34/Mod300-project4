# all the functions (and classes) for the project are defined in this file
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap, MWFaceOn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



def gen_milkyway():
    mw1 = MWFaceOn(
        radius=20 * u.kpc,
        unit=u.kpc,
        coord="galactocentric",
        annotation=True,
        figsize=(10, 8),
    )

    mw1.title = "Bird's Eyes View"

    mw1.scatter(8 * u.kpc, 0 * u.kpc, c="r", s=2)
    return mw1

    
def plt2rgbarr(fig):
    """
    A function to transform a matplotlib to a 3d rgb np.array 

    Input
    -----
    fig: matplotlib.figure.Figure
        The plot that we want to encode.        

    Output
    ------
    np.array(ndim, ndim, 3): A 3d map of each pixel in a rgb encoding (the three dimensions are x, y, and rgb)
    
    """
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr[:, :, :3]

def plot_mw_skymap(center,radius,background):

    skymap = MWSkyMap(center=center,
                    radius=radius,
                    background=background
                    )
    
    fig, ax = plt.subplots(figsize=(5,5))
    skymap.transform(ax)
    plt.title(f"{center} - Radius: {radius[0]} arcsec")
    plt.show()
    return fig, ax

def convert_to_rgb(figure):
    try:
        fig, ax = figure
        fig.canvas.draw()
    except:
        fig = figure
        fig.canvas.draw()
    img_1_arr = plt2rgbarr(fig)
    
    shape_image = img_1_arr.shape
    
    
    print(f"\nImage properties: ")    
    print(f"Image array shape:{shape_image}")
    print(f"Image array dtype: {img_1_arr.dtype}")
    print(f"Image array min, max values: {np.min(img_1_arr)}, {np.max(img_1_arr)}")
    print(f"First pixel RGB values: {img_1_arr[0, 0, :]}")
    
    
    return img_1_arr

def grey_encoding(img_array):
    grey = np.sum(img_array[: , : , :] * np.array([0.299, 0.587, 0.114]), axis=2)  # From RGB to grey
    x, y = [], []
    for ig, g in enumerate(grey):
        for ij, j in enumerate(g):
            if j > 230:
                x.append(ig)
                y.append(ij)
    return plt.scatter(x, y, s=0.1, c="grey")

def red_encoding(img_array):
    """
    Select pixels that are 'red' according to:
    - Red channel is bright
    - Red channel is significantly higher than green and blue
    """

    # Split channels
    R = img_array[..., 0].astype(float)
    G = img_array[..., 1].astype(float)
    B = img_array[..., 2].astype(float)

    # Normalize if floats in [0,1] vs uint8 in [0,255]
    if R.max() <= 1.0:
        scale = 1.0
        bright_thresh = 0.7         # ~ 180/255
        dominance_margin = 0.1      # R at least 0.1 higher than G,B
    else:
        scale = 255.0
        bright_thresh = 180.0       # less extreme than 230
        dominance_margin = 25.0     # R at least 25 higher than G,B

    # Create mask of "red" pixels
    red_mask = (
        (R > bright_thresh) &
        (R > G + dominance_margin) &
        (R > B + dominance_margin)
    )

    # Get coordinates of True values
    ys, xs = np.where(red_mask)  # ys = row indices, xs = col indices

    # Plot them in red
    return plt.scatter(ys, xs, s=0.1, c="r")

def blue_encoding(img_array):
    R = img_array[..., 0].astype(np.int16)
    G = img_array[..., 1].astype(np.int16)
    B = img_array[..., 2].astype(np.int16)

    bright_thresh = 180
    dominance_margin = 25

    blue_mask = (
        (B > bright_thresh) &
        (B > R + dominance_margin) &
        (B > G + dominance_margin)
    )

    ys, xs = np.where(blue_mask)
    return plt.scatter(xs, ys, s=0.1, c="b")

def kmeans_cluster_image(img_array, k=3, normalize=True, show_plot=True):  
    """  
    Perform K-means clustering on an RGB image.  
  
    Parameters  
    ----------  
    img_array : np.ndarray  
        RGB image array of shape (H, W, 3), dtype typically uint8 [0, 255].  
    k : int  
        Number of clusters for K-means.  
    normalize : bool  
        If True, scale RGB values to [0, 1] before clustering.  
    show_plot : bool  
        If True, display original image and clustering result side by side.  
  
    Returns  
    -------  
    label_image : np.ndarray  
        2D array of shape (H, W) with cluster labels in {0, 1, ..., k-1}.  
    kmeans : sklearn.cluster.KMeans  
        Fitted KMeans object (you can inspect cluster centers, etc.).  
    """  
  
    # ---- Step 1: Prepare data for K-means ----  
    h, w, c = img_array.shape  
    X = img_array.reshape(-1, 3).astype(float)   # (num_pixels, 3)  
  
    if normalize:  
        X /= 255.0  
  
    # ---- Step 2: Run K-means ----  
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")  
    kmeans.fit(X)  
  
    labels = kmeans.labels_                 # shape (num_pixels,)  
    label_image = labels.reshape(h, w)      # reshape back to image  
  
    print(f"K-means done with k={k}.")  
    print(f"Label image shape: {label_image.shape}")  
  
    # ---- Optional: Quick visualization ----  
    if show_plot:  
        # Define a simple color palette for clusters  
        # Repeat if k > len(base_colors)  
        base_colors = np.array([  
            [1.0, 0.0, 0.0],  # red  
            [0.0, 1.0, 0.0],  # green  
            [0.0, 0.0, 1.0],  # blue  
            [1.0, 1.0, 0.0],  # yellow  
            [1.0, 0.0, 1.0],  # magenta  
            [0.0, 1.0, 1.0],  # cyan  
        ])  
  
        # If k > base_colors, tile them  
        if k > len(base_colors):  
            repeats = int(np.ceil(k / len(base_colors)))  
            palette = np.tile(base_colors, (repeats, 1))[:k]  
        else:  
            palette = base_colors[:k]  
  
        # Map labels to colors  
        cluster_img = palette[label_image]  
  
        plt.figure(figsize=(10, 5))  
  
        plt.subplot(1, 2, 1)  
        plt.title("Original image")  
        plt.imshow(img_array)  
        plt.axis("off")  
  
        plt.subplot(1, 2, 2)  
        plt.title(f"K-means clustering (k={k})")  
        plt.imshow(cluster_img)  
        plt.axis("off")  
  
        plt.show()  
  
    return label_image, kmeans



def overlay_clusters_on_image(img_array, label_image, alpha=0.4, point_size=0.1):
    """
    Overlay K-means cluster assignments on top of the original image.

    Parameters
    ----------
    img_array : np.ndarray
        Original RGB image, shape (H, W, 3).
    label_image : np.ndarray
        2D array of cluster labels, shape (H, W), values in {0, ..., k-1}.
    alpha : float
        Transparency of the scatter points (0 = fully transparent, 1 = opaque).
    point_size : float
        Size of scatter points.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """

    h, w, _ = img_array.shape
    unique_labels = np.unique(label_image)
    k = len(unique_labels)

    # Simple color palette
    base_colors = np.array([
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 1.0],  # magenta
        [0.0, 1.0, 1.0],  # cyan
    ])

    if k > len(base_colors):
        repeats = int(np.ceil(k / len(base_colors)))
        palette = np.tile(base_colors, (repeats, 1))[:k]
    else:
        palette = base_colors[:k]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_array)
    ax.set_title(f"Clusters overlaid on Milky Way image (k={k})")
    ax.axis("off")

    # For each cluster, scatter its pixels
    for idx, label in enumerate(unique_labels):
        mask = (label_image == label)
        ys, xs = np.where(mask)   # ys = rows, xs = columns

        ax.scatter(
            xs, ys,
            s=point_size,
            c=[palette[idx]],
            alpha=alpha,
            label=f"Cluster {label}"
        )

    ax.legend(loc="lower right", fontsize=8)
    plt.show()

    return fig, ax

# convert an array of values into a dataset matrix
# copied from machinelearningmastery
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		dataX.append(dataset[i:(i + look_back), 0])
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)