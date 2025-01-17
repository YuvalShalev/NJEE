import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# For the progress bars:
from tqdm import trange, tqdm


###############################################################################
# 1. Utility Functions
###############################################################################

def convert_to_one_hot(y, num_classes):
    """
    Convert integer array to one-hot encoded tensor.
    """
    y_oh = np.eye(num_classes)[y.astype('int32')]
    return y_oh


def discretize(data, bins):
    """
    Discretize a 1D numpy array `data` into `bins` equiprobable bins.
    Returns:
        discrete: integer-coded array (same shape) with values in [0, bins-1]
        cutoffs : list of bin upper edges
    """
    sorted_data = np.sort(data)
    split = np.array_split(sorted_data, bins)
    cutoffs = [x[-1] for x in split[:-1]]

    discrete = np.digitize(data, cutoffs, right=True)
    return discrete, cutoffs


def discretize_batch(data, bins):
    """
    Discretize each column in 'data' into `bins` equiprobable bins.
    data shape: (N, D)
    """
    N, D = data.shape
    z_disc = np.zeros((N, D), dtype=np.int32)
    for d in range(D):
        z_disc[:, d], _ = discretize(data[:, d], bins)
    return z_disc


###############################################################################
# 2. Data Generation
###############################################################################

def making_cov(rho, dims):
    """
    Create a 2*dims x 2*dims covariance matrix with correlation = rho.
    """
    cov = np.zeros((2 * dims, 2 * dims))
    for i in range(dims):
        cov[i, i] = 1.0
        cov[i + dims, i + dims] = 1.0
        cov[i, i + dims] = rho
        cov[i + dims, i] = rho
    return cov


def generate_gaussian(rho, batch_size, dims):
    """
    Generate correlated Gaussian samples: [X1..Xd, Y1..Yd].
    """
    cov = making_cov(rho, dims)
    mean = np.zeros(2 * dims)
    z = np.random.multivariate_normal(mean=mean, cov=cov, size=batch_size)
    return z


###############################################################################
# 3. PyTorch MLP Model
###############################################################################

class MLPClassifier(nn.Module):
    """
    A 3-layer MLP that outputs `class_size` logits.
    """

    def __init__(self, input_dim, class_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, class_size)  # output logits
        )

    def forward(self, x):
        # x shape: [N, input_dim]
        return self.net(x)  # shape [N, class_size]


###############################################################################
# 4. Training Helpers
###############################################################################

def train_one_epoch(model, optimizer, x_np, y_np, batch_size=256):
    """
    A single epoch of training returns average loss.
    """
    device = next(model.parameters()).device

    x_t = torch.from_numpy(x_np).float().to(device)
    y_t = torch.from_numpy(y_np).long().to(device)

    dataset = TensorDataset(x_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    count = 0

    model.train()
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        count += batch_x.size(0)

    return total_loss / count


###############################################################################
# 5. Main NJEE Loop (with Progress Bar)
###############################################################################

def estimate_mi_torch(
        dims=20,
        bins=250,
        epochs=2000,
        batch_size=256,
        I_values=None
):
    """
    Estimate MI for multiple correlation values using a progress bar.
    I_values: array of target mutual information (nats).
    """
    if I_values is None:
        I_values = np.arange(2, 12, 2)  # default example

    # Convert I => rho
    r_lst = []
    for i in I_values:
        r = np.sqrt(1 - np.exp(-2 * i / dims))
        r_lst.append(r)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model lists for H(Y_j)
    model_lst = []
    opt_lst = []
    for m in range(dims):
        if m == 0:
            # For j=0, we do direct estimation from frequencies
            model_lst.append(None)
            opt_lst.append(None)
        else:
            mlp = MLPClassifier(m, bins).to(device)
            optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
            model_lst.append(mlp)
            opt_lst.append(optimizer)

    # Create model lists for H(Y_j | X, Y_1..Y_{j-1})
    model_lst_cond = []
    opt_lst_cond = []
    for m in range(dims):
        mlp = MLPClassifier(dims + m, bins).to(device)
        optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
        model_lst_cond.append(mlp)
        opt_lst_cond.append(optimizer)

    # Lists to track training progress
    H_y_lst = [[] for _ in range(dims)]
    H_yx_lst = [[] for _ in range(dims)]
    I_hat = []  # will store the running MI estimate

    # Outer loop: iterate over each target correlation
    for r_idx, r in enumerate(r_lst):
        print(f"=== Starting correlation r={r:.4f} for target I={I_values[r_idx]:.2f} ===")
        # Inner loop: epochs
        pbar = trange(epochs, desc=f"[r={r:.4f}]")  # Progress bar
        for epoch in pbar:
            z_0 = generate_gaussian(r, batch_size, dims)  # shape [batch_size, 2*dims]

            # For each dimension j in Y
            for j in range(dims):

                # 1) H(Y_j)
                if j == 0:
                    # discrete approach
                    y_j = z_0[:, j].reshape(-1, 1)
                    y_j_disc = discretize_batch(y_j, bins).flatten()

                    # freq-based plugin
                    unique_vals, counts = np.unique(y_j_disc, return_counts=True)
                    pvals = counts / (counts.sum() + 1e-20)
                    # naive plugin with Miller-Maddow correction
                    H_est = -np.sum(pvals * np.log(pvals + 1e-20)) + (len(unique_vals) - 1) / (2 * batch_size)
                    H_y_lst[j].append(H_est)
                else:
                    # use MLP for classification p(Y_j)
                    x_j = z_0[:, :j]
                    y_j = z_0[:, j].reshape(-1, 1)
                    y_j_disc = discretize_batch(y_j, bins).flatten()

                    loss_avg = train_one_epoch(model_lst[j], opt_lst[j], x_j, y_j_disc, batch_size)
                    H_y_lst[j].append(loss_avg)

                # 2) H(Y_j | X, Y_1..Y_{j-1})
                x_cond = z_0[:, :dims + j]
                y_cond = z_0[:, dims + j].reshape(-1, 1)
                y_cond_disc = discretize_batch(y_cond, bins).flatten()

                loss_avg_cond = train_one_epoch(model_lst_cond[j],
                                                opt_lst_cond[j],
                                                x_cond, y_cond_disc,
                                                batch_size)
                H_yx_lst[j].append(loss_avg_cond)

            # After training all j, compute the last MI estimate
            H_y = sum(h[-1] for h in H_y_lst)  # sum of final appended entropies
            H_yx = sum(hx[-1] for hx in H_yx_lst)
            I_hat.append(H_y - H_yx)

    return I_hat, I_values


###############################################################################
# 6. Running & Plotting
###############################################################################

if __name__ == "__main__":
    dims = 20
    bins = 250
    epochs = 1500
    batch_size = 256
    I_values = np.arange(2, 12, 2)

    print("Starting NJEE estimation")
    I_hat, I_vals = estimate_mi_torch(
        dims=dims,
        bins=bins,
        epochs=epochs,
        batch_size=batch_size,
        I_values=I_values
    )

    # Convert the results to a NumPy array
    I_hat_arr = np.array(I_hat)  # length = len(I_values) * epochs

    # True MI repeated for plotting
    I_real = np.hstack([np.repeat(i, epochs) for i in I_vals])

    # Smooth with an exponential moving average if you like
    df_hat = pd.DataFrame(I_hat_arr, columns=['MI_hat'])
    EMA_SPAN = 200
    mi_smooth = df_hat.ewm(span=EMA_SPAN).mean()

    plt.figure(figsize=(8, 5))
    plt.plot(I_real, 'k', label='True MI')
    plt.plot(mi_smooth, 'r', label='MI_hat (smoothed)')
    plt.ylabel('Mutual Information (nats)')
    plt.xlabel('Batch number')
    plt.title('NJEE-based MI estimation')
    plt.legend()
    plt.show()
