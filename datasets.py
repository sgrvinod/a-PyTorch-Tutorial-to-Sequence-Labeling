from torch.utils.data import Dataset
import torch


class WCDataset(Dataset):
    """
    PyTorch Dataset for the LM-LSTM-CRF model. To be used by a PyTorch DataLoader to feed batches to the model.
    """

    def __init__(self, wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths):
        """
        :param wmaps: padded encoded word sequences
        :param cmaps_f: padded encoded forward character sequences
        :param cmaps_b: padded encoded backward character sequences
        :param cmarkers_f: padded forward character markers
        :param cmarkers_b: padded backward character markers
        :param tmaps: padded encoded tag sequences (indices in unrolled CRF scores)
        :param wmap_lengths: word sequence lengths
        :param cmap_lengths: character sequence lengths
        """
        self.wmaps = wmaps
        self.cmaps_f = cmaps_f
        self.cmaps_b = cmaps_b
        self.cmarkers_f = cmarkers_f
        self.cmarkers_b = cmarkers_b
        self.tmaps = tmaps
        self.wmap_lengths = wmap_lengths
        self.cmap_lengths = cmap_lengths

        self.data_size = self.wmaps.size(0)

    def __getitem__(self, i):
        return self.wmaps[i], self.cmaps_f[i], self.cmaps_b[i], self.cmarkers_f[i], self.cmarkers_b[i], self.tmaps[i], \
               self.wmap_lengths[i], self.cmap_lengths[i]

    def __len__(self):
        return self.data_size
