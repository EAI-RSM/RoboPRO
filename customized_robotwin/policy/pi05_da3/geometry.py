import numpy as np


def as_homogeneous(ext):
    if not isinstance(ext, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(ext)}")
    if ext.shape[-2:] == (3, 4):
        ones = np.zeros_like(ext[..., :1, :4])
        ones[..., 0, 3] = 1.0
        return np.concatenate([ext, ones], axis=-2)
    elif ext.shape[-2:] == (4, 4):
        return ext
    else:
        raise ValueError(f"Invalid shape for extrinsics: {ext.shape}")


def homogenize(x, points=True):
    last_row_fn = np.ones_like if points else np.zeros_like
    last_row = last_row_fn(x[..., :1])
    return np.concatenate([x, last_row], axis=-1)


def transpose_last_two_axes(arr):
    """
    for np < 2
    """
    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))
    # swap the last two
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return arr.transpose(axes)


def affine_inverse_np(A: np.ndarray):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return np.concatenate(
        [
            np.concatenate([transpose_last_two_axes(R), -transpose_last_two_axes(R) @ T], axis=-1),
            P,
        ],
        axis=-2,
    )


def transform(mat, pts):
    return np.einsum("...ij,...j->...i", mat, pts)


def pixel_space_to_camera_space(pixel_pts, depth, K):
    """
    Args:
        pixel_pts: (H, W, 2)
        depth: (N, H, W, 1)
        K: (N, 3, 3)
    Returns:
        cam_pts: (N, H, W, 3)
    """
    pixel_pts = homogenize_points(pixel_pts)  # (H,W,3)

    K_inv = inverse_intrinsic_matrix(K)

    cam = np.einsum("v i j , h w j-> v h w i", K_inv, pixel_pts)
    cam = cam * depth

    return cam


def camera_space_to_world_space(cam_pts, c2w):
    """
    Args:
        cam_pts: (V, H, W, 3)
        c2w: (V, 4, 4)
    Returns:
        world_pts: (V, H, W, 3)
    """
    cam_h = homogenize_points(cam_pts)  # (V, H, W, 4)

    # broadcast matmul
    world = np.einsum("v i j , v h w j-> v h w i", c2w, cam_h)
    return world[..., :3]


def unproject_depth(depth, K, c2w):
    """
    Args:
        depth: (V, H, W, 1)
        K: (V, 3, 3)
        c2w: (V, 3, 4)
    Returns:
        world_pts: (V, H, W, 3)
    """

    H, W = depth.shape[-3], depth.shape[-2]
    c2w = as_homogeneous(c2w)  # (V, 4, 4)

    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")  # (h, w), (h, w)
    pixel = np.stack([x, y], axis=-1)  # (..., H, W, 2)

    cam = pixel_space_to_camera_space(pixel, depth, K)  # (..., H, W, 3)
    world = camera_space_to_world_space(cam, c2w)  # (..., H, W, 3)

    return world