import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math


# ------------------------------------------------------------
# 1.  Zipf Data Generator & Entropy Computation
# ------------------------------------------------------------
def zipf_dist(alpha, N, size, seed=0):
    """
    Draw 'size' samples from a Zipf(alpha) truncated to {1,...,N}.
    Returns:
        data: (size, 1) array of integers in [1..N].
        weights: the pmf (length N) of the truncated Zipf.
    """
    # For reproducibility
    np.random.seed(seed)

    x = np.arange(1, N+1, dtype=float)
    weights = x ** (-alpha)
    weights /= weights.sum()

    # Use a discrete distribution sampler from SciPy-like logic
    # We’ll do manual sampling here in pure NumPy
    # because PyTorch environment may not have scipy.stats.
    cdf = np.cumsum(weights)
    rnd = np.random.rand(size)
    data = np.searchsorted(cdf, rnd) + 1   # +1 b/c 0-based index => 1-based symbol

    return data.reshape(-1, 1), weights


def harmonic(n, alpha):
    # Simple harmonic sum: sum_{i=1}^{n-1} (1 / i^alpha)
    a = 0.
    for i in range(1, n):
        a += 1. / (i**alpha)
    return a


def zipf_entropy(alphabet, alpha):
    """
    Theoretical Zipf entropy in *nats* because we use np.log (natural log).
    H_zipf = -(1/c) sum_{x=1 to n} [ p(x)*ln( p(x)/c ) ]
    where p(x) ~ x^{-alpha}, and c = harmonic(n, alpha).
    """
    p = np.arange(1, alphabet, dtype='float')**(-alpha)
    c = harmonic(alphabet, alpha)
    H_zipf = -(1.0 / c) * np.sum(p * np.log(p / c))
    return H_zipf


# ------------------------------------------------------------
# 2.  Convert data => shift for next-bit labels
# ------------------------------------------------------------
def symbols_to_bits(data, dims=25):
    """
    data shape: (num_samples, 1), each in [1..alphabet].
    Return shape: (num_samples, dims), each row is a 25-bit representation.
    """
    vf = np.vectorize(np.binary_repr)
    # 1) Convert to binary string of length `dims`
    data_bin = vf(data, width=dims)   # shape (num_samples, 1) of strings
    # 2) Turn each string into a list of 0/1 ints
    out = []
    for bits in data_bin[:, 0]:       # each is a string of length `dims`
        out.append([int(b) for b in bits])
    return np.array(out, dtype=np.float32)


def make_shifted_labels(bits_array):
    """
    bits_array shape: (N, seq_length).
    We want to predict bit[i+1] from bit[i], so each position's label is the *next* bit.

    - We'll feed the entire sequence to the LSTM,
    - The LSTM outputs a probability for each of the 2 classes (0 or 1) at each time step,
    - We shift the 'target' by 1 to align with the LSTM outputs.

    For the final bit, you might assign a dummy label (e.g., 0).
    """
    shifted_labels = np.zeros_like(bits_array)
    # Shift everything left by 1, final bit = 0
    shifted_labels[:, :-1] = bits_array[:, 1:]
    shifted_labels[:, -1] = 0  # for the very last bit
    return shifted_labels


# ------------------------------------------------------------
# 3.  Define a simple LSTM-based classifier
# ------------------------------------------------------------
class BitPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(BitPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # 2 classes: 0 or 1

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_size=1)
        returns logits of shape (batch_size, seq_length, 2)
        """
        lstm_out, _ = self.lstm(x)       # (B, L, hidden_size)
        logits = self.fc(lstm_out)       # (B, L, 2)
        return logits


# ------------------------------------------------------------
# 4.  Main Script
# ------------------------------------------------------------
def main():
    # -----------------------------
    # Hyperparameters
    # -----------------------------
    source = 'zipf'
    alpha = 1.0
    alphabet = 10**8
    size = 1000
    dims = 30
    batch_size = 64
    epochs = 200
    seed = 0   # for reproducibility

    # -----------------------------
    # Generate data & compute true entropy
    # -----------------------------
    data_int, _ = zipf_dist(alpha, alphabet, size, seed=seed)   # shape (size, 1)
    H_true = zipf_entropy(alphabet, alpha)

    # Convert to bits
    data_bits     = symbols_to_bits(data_int, dims=dims)          # shape (size, dims)
    # Create target: shift bits by 1
    labels_bits   = make_shifted_labels(data_bits)                # shape (size, dims)

    x_data = data_bits.reshape(size, dims, 1)    # (N, dims, 1)
    y_data = labels_bits                         # (N, dims)

    x_tensor = torch.from_numpy(x_data)          # shape (N, dims, 1)
    y_tensor = torch.from_numpy(y_data).long()   # shape (N, dims), each entry in {0,1}

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # -----------------------------
    # Define model, loss, optimizer
    # -----------------------------
    model = BitPredictor(input_size=1, hidden_size=50, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)  # shape (batch_size, seq_length, 2)
            batch_size_i, seq_len, _ = logits.shape
            logits_flat = logits.reshape(batch_size_i*seq_len, 2)  # (N*L,2)
            by_flat = by.reshape(batch_size_i*seq_len)         # (N*L)

            loss = criterion(logits_flat, by_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size_i

        # Optional: print train loss
        # avg_loss = total_loss / size
        # if (epoch+1) % 50 == 0:
        #     print(f"Epoch {epoch+1:03d}/{epochs}, Loss={avg_loss:.4f}")

    # -----------------------------
    # Compute final cross-entropy estimate
    # -----------------------------
    model.eval()
    with torch.no_grad():
        # Get predicted probabilities for the entire dataset
        logits_all = model(x_tensor)  # shape (size, 25, 2)
        # Convert logits to probabilities
        probs_all = torch.softmax(logits_all, dim=-1)  # (size, 25, 2)

        # Extract the "correct class" probability at each position
        # y_tensor: (size, dims) in {0,1}
        # gather along last dim => shape (size, 25)
        correct_probs = probs_all.gather(dim=2, index=y_tensor.unsqueeze(-1)).squeeze(-1)

        # -log p for each bit => sum over dims => average over dataset
        neg_log_probs = -torch.log(correct_probs + 1e-12)
        ce_per_seq = neg_log_probs.sum(dim=1)    # sum over seq_length => shape (size,)
        CE_mean = ce_per_seq.mean().item()    # final average

    # "First-bit" plug-in correction
    y_np = data_bits[:, 0]  # first bit across all samples
    p_1 = (y_np.sum() + 1e-5) / float(size)
    p_2 = 1.0 - p_1
    H_1 = -(p_1 * math.log(p_1) + p_2 * math.log(p_2))

    # Final estimate
    H_estimated = CE_mean + H_1

    # -----------------------------
    # Print the result
    # -----------------------------
    print(f"H estimated  {H_estimated:.4f}")
    print(f"H true       {H_true:.4f}")


if __name__ == "__main__":
    main()
