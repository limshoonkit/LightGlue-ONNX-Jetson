"""
2D visualization primitives based on Matplotlib.
1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

import matplotlib
import matplotlib.cm
import matplotlib.patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, BGR (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, list | tuple):
        cmaps = [cmaps] * n

    ratios = [i.shape[1] / i.shape[0] for i in imgs] if adaptive else [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    ax: list
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios})  # type: ignore
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i][..., ::-1], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)

def plot_extractor_only(images, batch_size, kpts, num_kpts, extractor_name="Extractor"):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    output_images = []

    h, w = images[0].shape[:2]

    for i in range(batch_size):
        # Get the valid keypoints for this image
        num = int(num_kpts[i]) if hasattr(num_kpts, "__iter__") else int(num_kpts)
        kpts_i = kpts[i, :num, :]

        if kpts_i.size == 0:
            kpts_pixel = np.empty((0, 2))
        else:
            # If most points fall inside [-1, 1], perform denormalization to get pixel coords
            if np.all((kpts_i >= -1.1) & (kpts_i <= 1.1)):
                wh = np.array([w - 1, h - 1], dtype=np.float32)
                kpts_pixel = (kpts_i + 1) / 2 * wh
            else:
                kpts_pixel = kpts_i  # already in pixel coords

        # Convert keypoints to OpenCV's format
        cv_kpts = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in kpts_pixel]

        # Draw keypoints on the image
        color = (0, 255, 0) if "superpoint" in extractor_name.lower() else (255, 0, 0)
        img_with_kpts = cv2.drawKeypoints(images[i], cv_kpts, None, color=color)
        output_images.append(img_with_kpts)

    # Combine images side-by-side
    combined_image = np.hstack(output_images)

    # Display
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.title(f"{extractor_name} Keypoints")
    plt.axis("off")
    plt.show()


def plot_keypoints(kpts, colors="lime", ps=4, axes=None, a=1.0):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(a, list):
        a = [a] * len(kpts)
    if axes is None:
        axes = plt.gcf().axes
    for ax, k, c, alpha in zip(axes, kpts, colors, a, strict=False):
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)

def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, a=1.0, labels=None, axes=None):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()  # type: ignore  # noqa: NPY002
    elif len(color) > 0 and not isinstance(color[0], tuple | list):
        color = [color] * len(kpts0)

    if lw > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def add_text(idx, text, pos=(0.01, 0.99), fs=15, color="w", lcolor="k", lwidth=2, ha="left", va="top"):
    ax = plt.gcf().axes[idx]
    t = ax.text(*pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes)
    if lcolor is not None:
        t.set_path_effects([path_effects.Stroke(linewidth=lwidth, foreground=lcolor), path_effects.Normal()])


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
