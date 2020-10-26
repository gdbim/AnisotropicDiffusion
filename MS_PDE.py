import numpy as np


def pack(a, b):
    """
    a and b are 2 arrays defined on the domain of size (nx,ny). Flatten will convert them into 2 vectors
    of sizes (nx*ny). Finally, concatenate will build a vector of size (2*nx*ny) containing the unknowns of both a and b.
    
    `flatten`: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    `concatenate`: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    """
    return np.concatenate((a.flatten(), b.flatten()))

def unpack(X, steps):
    """
    X is a vector of size (2*nx*ny). The first nx*ny elements belong to a and the nx*ny following belong to b.
    X[:n] will be a vector of size (nx*ny). Reshape it will convert it to an array of size (nx,ny).
    
    `reshape`: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    `prod`: https://numpy.org/doc/stable/reference/generated/numpy.prod.html
    """
    n = np.prod(steps)
    return X[:n].reshape(steps), X[n:].reshape(steps)


# Diffusion part of the MS model
def Div(v, r, dx2, dy2, nx, ny, scar):
    v[scar] = 0

    # d will be the result of the laplacian operator
    # We will fill its nx*ny elements
    d = np.empty((nx, ny))
    
    # Start with finite differences in the interior of the domain
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            d[i, j] = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx2 + r * (
                v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]
            ) / dy2

    # Boundary conditions: shifted scheme
    # left
    j = 0
    for i in range(1, nx - 1):
        d[i, j] = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx2 + r * (
            v[i, j + 2] - 2 * v[i, j + 1] + v[i, j]
        ) / dy2

    # right
    j = ny - 1
    for i in range(1, nx - 1):
        d[i, j] = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx2 + r * (
            v[i, j] - 2 * v[i, j - 1] + v[i, j - 2]
        ) / dy2

    # top
    i = 0
    for j in range(1, ny - 1):
        d[i, j] = (v[i + 2, j] - 2 * v[i + 1, j] + v[i, j]) / dx2 + r * (
            v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]
        ) / dy2

    # bottom
    i = nx - 1
    for j in range(1, ny - 1):
        d[i, j] = (v[i, j] - 2 * v[i - 1, j] + v[i - 2, j]) / dx2 + r * (
            v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]
        ) / dy2

    # Corners
    i, j = 0, 0
    d[i, j] = (v[i + 2, j] - 2 * v[i + 1, j] + v[i, j]) / dx2 + r * (
        v[i, j + 2] - 2 * v[i, j + 1] + v[i, j]
    ) / dy2

    i, j = nx - 1, 0
    d[i, j] = (v[i, j] - 2 * v[i - 1, j] + v[i - 2, j]) / dx2 + r * (
        v[i, j + 2] - 2 * v[i, j + 1] + v[i, j]
    ) / dy2

    i, j = 0, ny - 1
    d[i, j] = (v[i + 2, j] - 2 * v[i + 1, j] + v[i, j]) / dx2 + r * (
        v[i, j] - 2 * v[i, j - 1] + v[i, j - 2]
    ) / dy2

    i, j = nx - 1, ny - 1
    d[i, j] = (v[i, j] - 2 * v[i - 1, j] + v[i - 2, j]) / dx2 + r * (
        v[i, j] - 2 * v[i, j - 1] + v[i, j - 2]
    ) / dy2

    d[scar] = 0
    return d
