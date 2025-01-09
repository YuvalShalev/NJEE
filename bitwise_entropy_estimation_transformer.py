import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim

##############################################################################
# 1) Simulated Data Generators (same as original)
##############################################################################


def zipf_dist(alpha, N, size):
    """
    Draw 'size' samples from a truncated Zipf(alpha) over {1,...,N}.
    Returns: data of shape (size,1) and the pmf weights of length N.
    """
    x = np.arange(1, N+1, dtype='float')
    weights = x ** (-alpha)
    weights /= weights.sum()

    # Sample: (size,)
    cdf = np.cumsum(weights)
    rnd = np.random.rand(size)
    data = np.searchsorted(cdf, rnd) + 1  # +1 for 1-based indexing

    return data.reshape(-1,1), weights


def harmonic(n, alpha):
    """Compute sum_{i=1}^{n-1} of 1/(i^alpha)."""
    a = 0.0
    for i in range(1, n):
        a += 1.0 / (i**alpha)
    return a


def zipf_entropy(alphabet, alpha):
    """
    Compute the theoretical Zipf entropy in nats.
    H = -(1/c) sum_{x=1 to n} [ p(x) * ln( p(x)/c ) ],  p(x) ~ x^{-alpha}
    """
    p = np.arange(1, alphabet, dtype='float')**(-alpha)
    c = harmonic(alphabet, alpha)
    H = -(1.0 / c) * np.sum(p * np.log(p / c))
    return H


def geom_entropy(weights):
    """Optional for geometric distributions, if needed."""
    return -np.sum(weights * np.log(weights))


##############################################################################
# 2) Utilities: convert data to bits, shift for next-bit labels
##############################################################################
def convert_to_one_hot(y, dict_size=None):
    """
    Convert 1D integer array y into one-hot matrix of shape (len(y), dict_size).
    Not used directly here, but left for reference.
    """
    if dict_size is None:
        dict_size = np.unique(y).shape[0]
    y_hot = np.eye(dict_size)[y.astype('int32')]
    return y_hot


def symbols_to_bits(data, dims=25):
    """
    Convert integer symbols in data (shape: (N,1)) to binary strings of length `dims`.
    Then parse each string into a list of 0/1.
    Returns array of shape (N, dims).
    """
    vf = np.vectorize(np.binary_repr)
    bin_strs = vf(data, width=dims)  # shape (N,1) of strings
    arr = []
    for s in bin_strs[:, 0]:
        arr.append([int(b) for b in s])
    return np.array(arr, dtype=np.float32)


def shift_bits(x_bits):
    """
    Next-bit prediction: the label at position t is the bit at position t+1.
    So we shift left by 1, and set the last bit=0 for the label.
    x_bits: shape (N, seq_len)
    Returns: shape (N, seq_len), integer in {0,1}.
    """
    y_bits = np.zeros_like(x_bits)
    y_bits[:, :-1] = x_bits[:, 1:]
    y_bits[:, -1]  = 0
    return y_bits


##############################################################################
# 3) Define a Transformer Model (instead of RNN)
##############################################################################
class TransformerBitPredictor(nn.Module):
    """
    In the original NJEE paper we used RNN. We can also use more efficient and powerful architecture such as a causal
    Transformer that maps a sequence of bits (0/1) to next-bit predictions.
    We'll embed bits, apply positional encoding, then a Transformer encoder,
    then output logits for each time step (0 vs 1).
    """
    def __init__(self, seq_length=25, d_model=32, nhead=4, num_layers=2, dim_ff=64):
        super().__init__()
        # We'll embed bits {0,1} => d_model
        self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        # A simple positional encoding so the Transformer knows positions
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=seq_length)

        # Standard TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_ff,
                                                   dropout=0.1,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier: from d_model => 2 classes
        self.fc_out = nn.Linear(d_model, 2)

        # Create a causal mask (subsequent mask) for sequences of length `seq_length`
        # so the model can't peek at future bits.
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length),
                 each entry in {0,1} is the bit at that time-step.

        returns logits shape: (batch_size, seq_length, 2)
        """
        # 1) Convert bits to embeddings => (B, L, d_model)
        emb = self.embedding(x.long())
        emb = self.pos_encoding(emb)

        # 2) Pass through the TransformerEncoder, with a causal mask
        out = self.transformer(emb, mask=self.causal_mask)

        # 3) Convert final hidden states to logits (2 classes)
        logits = self.fc_out(out)  # shape (B, L, 2)
        return logits


##############################################################################
# 3a) Positional Encoding (standard sine/cosine)
##############################################################################
class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))
        # fill even and odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (B, L, d_model). We'll add positional enc by broadcast.
        """
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return x


##############################################################################
# 4) Training + Entropy Estimation (similar to your original RNN code)
##############################################################################
def main():
    # -----------------------------
    # Hyperparameters
    # -----------------------------
    source = 'zipf'
    alpha = 1.0
    alphabet = 10**8
    size = 1000
    seq_length = 30
    epochs = 200
    batch_size = 64
    seed = 0   # for reproducibility

    # Fix random seeds (optional)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -----------------------------
    # Generate data from Zipf
    # -----------------------------
    if source == 'zipf':
        data_int, _ = zipf_dist(alpha, alphabet, size)
        H_true = zipf_entropy(alphabet, alpha)
    else:
        raise ValueError("Only 'zipf' example is shown here.")

    # Convert integer symbols => 25-bit => shift
    x_bits = symbols_to_bits(data_int, dims=seq_length)  # shape (N, 25)
    y_bits = shift_bits(x_bits)                          # shape (N, 25)

    # Turn them into torch Tensors
    X = torch.tensor(x_bits, dtype=torch.long)  # (N,25), bits => {0,1}
    Y = torch.tensor(y_bits, dtype=torch.long)  # (N,25), bits => {0,1}

    # For training in mini-batches
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # -----------------------------
    # Define the Transformer-based model
    # -----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerBitPredictor(seq_length=seq_length, d_model=32, nhead=4, num_layers=2, dim_ff=64)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            bx = bx.to(device)  # shape (B,25)
            by = by.to(device)  # shape (B,25)

            optimizer.zero_grad()
            logits = model(bx)          # (B, 25, 2)
            # Flatten for cross-entropy
            logits_flat = logits.reshape(-1, 2)     # (B*25, 2)
            by_flat = by.reshape(-1)           # (B*25)
            loss = criterion(logits_flat, by_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
    # -----------------------------
    # Estimate Cross-Entropy
    # -----------------------------
    model.eval()
    with torch.no_grad():
        # We'll compute -log p(correct) over the entire dataset
        all_neglog = []
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)                 # (B,25,2)
            probs = torch.softmax(logits, -1) # (B,25,2)

            # gather probability of the correct bit
            correct_p = probs.gather(2, by.unsqueeze(-1)).squeeze(-1)  # (B,25)
            neglog    = -torch.log(correct_p + 1e-12)  # avoid log(0)
            seq_sum   = neglog.sum(dim=1)              # sum over 25
            all_neglog.extend(seq_sum.cpu().numpy())

        CE = np.mean(all_neglog)  # average over all sequences

    # "First-bit" correction (same logic as your RNN code)
    first_bit = x_bits[:, 0]  # shape (N,)
    p_1 = (np.sum(first_bit) + 1e-5) / float(size)
    H_1 = -(p_1 * math.log(p_1) + (1 - p_1)*math.log(1 - p_1))

    # Final estimated entropy
    H_est = CE + H_1

    # -----------------------------
    # Print results
    # -----------------------------
    print(f"H estimated {H_est:.4f}")
    print(f"H true      {H_true:.4f}")


##############################################################################
# 5) Entry Point
##############################################################################
if __name__ == "__main__":
    main()
