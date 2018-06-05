import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Create a tensor with variable length sequences and pads (25)
seqs = torch.LongTensor([[0, 1, 2, 3, 25, 25, 25],
                         [4, 5, 25, 25, 25, 25, 25],
                         [6, 7, 8, 9, 10, 11, 25]])

# Store lengths of the actual sequences, ignoring padding
# These are the points up to which we want the RNN to process the sequence
seq_lens = torch.LongTensor([4, 2, 6])

# Sort by decreasing lengths
seq_lens, sort_ind = seq_lens.sort(dim=0, descending=True)
seqs = seqs[sort_ind]

# Create an embedding layer, with 0 vectors for the pads
embeds = nn.Embedding(26, 10, padding_idx=25)

# Create an LSTM layer
lstm = nn.LSTM(10, 50, bidirectional=False, batch_first=True)

# WITHOUT DYNAMIC BATCHING

embeddings = embeds(seqs)
out_static, _ = lstm(embeddings)

# The number of timesteps in the output will be the same as the total padded timesteps in the input,
# since the LSTM computed over the pads
assert out_static.size(1) == embeddings.size(1)

# Look at the output at a timestep that we know is a pad
print(out_static[1, -1])

# WITH DYNAMIC BATCHING

# Pack the sequence
packed_seqs = pack_padded_sequence(embeddings, seq_lens.tolist(), batch_first=True)

# To execute the LSTM over only the valid timesteps
out_dynamic, _ = lstm(packed_seqs)

# Use the inverse function to re-pad it
out_dynamic, lens = pad_packed_sequence(out_dynamic, batch_first=True)

# Note that since we re-padded it, the total padded timesteps will be the length of the longest sequence (6)
assert out_dynamic.size(1) != embeddings.size(1)
print(out_dynamic.shape)

# Look at the output at a timestep that we know is a pad
print(out_dynamic[1, -1])

# It's all zeros!

#########################################################

# So, what does pack_padded_sequence do?
# It removes pads, flattens by timestep, and keeps track of effective batch_size at each timestep

# The RNN computes only on the effective batch size "b_t" at each timestep
# This is why we sort - so the top "b_t" rows at timestep "t" are aligned with the top "b_t" outputs from timestep "t-1"

# Consider the original encoded sequences (sorted)
print(seqs)

# Let's pack it
packed_seqs = pack_padded_sequence(seqs, seq_lens, batch_first=True)

# The result of pack_padded_sequence() is a tuple containing the flattened tensor and the effective batch size at each timestep
# Here's the flattened tensor with pads removed
print(packed_seqs[0])
# You can see it's flattened timestep-wise
# Since pads are removed, the total datapoints are equal to the number of valid timsteps
assert packed_seqs[0].size(0) == sum(seq_lens.tolist())

# Here's the effective batch size at each timestep
print(packed_seqs[1])
# If you look at the original encoded sequences, you can see this is true
print(seqs)

