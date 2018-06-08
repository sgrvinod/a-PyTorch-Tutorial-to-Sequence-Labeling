import time
import torch
import torch.optim as optim
import os
import sys
from models import LM_LSTM_CRF, ViterbiLoss
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import WCDataset
from inference import ViterbiDecoder
from sklearn.metrics import f1_score

# Data parameters
task = 'ner'  # tagging task, to choose column in CoNLL 2003 dataset
train_file = './datasets/eng.train'  # path to training data
val_file = './datasets/eng.testa'  # path to validation data
test_file = './datasets/eng.testb'  # path to test data
emb_file = './embeddings/glove.6B.100d.txt'  # path to pre-trained word embeddings
min_word_freq = 5  # threshold for word frequency
min_char_freq = 1  # threshold for character frequency
caseless = True  # lowercase everything?
expand_vocab = True  # expand model's input vocabulary to the pre-trained embeddings' vocabulary?

# Model parameters
char_emb_dim = 30  # character embedding size
with open(emb_file, 'r') as f:
    word_emb_dim = len(f.readline().split(' ')) - 1  # word embedding size
word_rnn_dim = 300  # word RNN size
char_rnn_dim = 300  # character RNN size
char_rnn_layers = 1  # number of layers in character RNN
word_rnn_layers = 1  # number of layers in word RNN
highway_layers = 1  # number of layers in highway network
dropout = 0.5  # dropout
fine_tune_word_embeddings = False  # fine-tune pre-trained word embeddings?

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 10  # batch size
lr = 0.015  # learning rate
lr_decay = 0.05  # decay learning rate by this amount
momentum = 0.9  # momentum
workers = 1  # number of workers for loading data in the DataLoader
epochs = 200  # number of epochs to run without early-stopping
grad_clip = 5.  # clip gradients at this value
print_freq = 100  # print training or validation status every __ batches
best_f1 = 0.  # F1 score to start with
checkpoint = None  # path to model checkpoint, None if none

tag_ind = 1 if task == 'pos' else 3  # choose column in CoNLL 2003 dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    Training and validation.
    """
    global best_f1, epochs_since_improvement, checkpoint, start_epoch, word_map, char_map, tag_map

    # Read training and validation data
    train_words, train_tags = read_words_tags(train_file, tag_ind, caseless)
    val_words, val_tags = read_words_tags(val_file, tag_ind, caseless)

    # Initialize model or load checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        word_map = checkpoint['word_map']
        lm_vocab_size = checkpoint['lm_vocab_size']
        tag_map = checkpoint['tag_map']
        char_map = checkpoint['char_map']
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']
    else:
        word_map, char_map, tag_map = create_maps(train_words + val_words, train_tags + val_tags, min_word_freq,
                                                  min_char_freq)  # create word, char, tag maps
        embeddings, word_map, lm_vocab_size = load_embeddings(emb_file, word_map,
                                                              expand_vocab)  # load pre-trained embeddings

        model = LM_LSTM_CRF(tagset_size=len(tag_map),
                            charset_size=len(char_map),
                            char_emb_dim=char_emb_dim,
                            char_rnn_dim=char_rnn_dim,
                            char_rnn_layers=char_rnn_layers,
                            vocab_size=len(word_map),
                            lm_vocab_size=lm_vocab_size,
                            word_emb_dim=word_emb_dim,
                            word_rnn_dim=word_rnn_dim,
                            word_rnn_layers=word_rnn_layers,
                            dropout=dropout,
                            highway_layers=highway_layers).to(device)
        model.init_word_embeddings(embeddings.to(device))  # initialize embedding layer with pre-trained embeddings
        model.fine_tune_word_embeddings(fine_tune_word_embeddings)  # fine-tune
        optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)

    # Loss functions
    lm_criterion = nn.CrossEntropyLoss().to(device)
    crf_criterion = ViterbiLoss(tag_map).to(device)

    # Since the language model's vocab is restricted to in-corpus indices, encode training/val with only these!
    # word_map might have been expanded, and in-corpus words eliminated due to low frequency might still be added because
    # they were in the pre-trained embeddings
    temp_word_map = {k: v for k, v in word_map.items() if v <= word_map['<unk>']}
    train_inputs = create_input_tensors(train_words, train_tags, temp_word_map, char_map,
                                        tag_map)
    val_inputs = create_input_tensors(val_words, val_tags, temp_word_map, char_map, tag_map)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(WCDataset(*train_inputs), batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(WCDataset(*val_inputs), batch_size=batch_size, shuffle=True,
                                             num_workers=workers, pin_memory=False)

    # Viterbi decoder (to find accuracy during validation)
    vb_decoder = ViterbiDecoder(tag_map)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              lm_criterion=lm_criterion,
              crf_criterion=crf_criterion,
              optimizer=optimizer,
              epoch=epoch,
              vb_decoder=vb_decoder)

        # One epoch's validation
        val_f1 = validate(val_loader=val_loader,
                          model=model,
                          crf_criterion=crf_criterion,
                          vb_decoder=vb_decoder)

        # Did validation F1 score improve?
        is_best = val_f1 > best_f1
        best_f1 = max(val_f1, best_f1)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_f1, word_map, char_map, tag_map, lm_vocab_size, is_best)

        # Decay learning rate every epoch
        adjust_learning_rate(optimizer, lr / (1 + (epoch + 1) * lr_decay))


def train(train_loader, model, lm_criterion, crf_criterion, optimizer, epoch, vb_decoder):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param lm_criterion: cross entropy loss layer
    :param crf_criterion: viterbi loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    :param vb_decoder: viterbi decoder (to decode and find F1 score)
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    ce_losses = AverageMeter()  # cross entropy loss
    vb_losses = AverageMeter()  # viterbi loss
    f1s = AverageMeter()  # f1 score

    start = time.time()

    # Batches
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths) in enumerate(
            train_loader):

        data_time.update(time.time() - start)

        max_word_len = max(wmap_lengths.tolist())
        max_char_len = max(cmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(device)
        cmaps_f = cmaps_f[:, :max_char_len].to(device)
        cmaps_b = cmaps_b[:, :max_char_len].to(device)
        cmarkers_f = cmarkers_f[:, :max_word_len].to(device)
        cmarkers_b = cmarkers_b[:, :max_word_len].to(device)
        tmaps = tmaps[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)
        cmap_lengths = cmap_lengths.to(device)

        # Forward prop.
        crf_scores, lm_f_scores, lm_b_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, __ = model(cmaps_f,
                                                                                                             cmaps_b,
                                                                                                             cmarkers_f,
                                                                                                             cmarkers_b,
                                                                                                             wmaps,
                                                                                                             tmaps,
                                                                                                             wmap_lengths,
                                                                                                             cmap_lengths)

        # LM loss

        # We don't predict the next word at the pads or <end> tokens
        # We will only predict at [dunston, checks, in] among [dunston, checks, in, <end>, <pad>, <pad>, ...]
        # So, prediction lengths are word sequence lengths - 1
        lm_lengths = wmap_lengths_sorted - 1
        lm_lengths = lm_lengths.tolist()

        # Remove scores at timesteps we won't predict at
        # pack_padded_sequence is a good trick to do this (see dynamic_rnn.py, where we explore this)
        lm_f_scores, _ = pack_padded_sequence(lm_f_scores, lm_lengths, batch_first=True)
        lm_b_scores, _ = pack_padded_sequence(lm_b_scores, lm_lengths, batch_first=True)

        # For the forward sequence, targets are from the second word onwards, up to <end>
        # (timestep -> target) ...dunston -> checks, ...checks -> in, ...in -> <end>
        lm_f_targets = wmaps_sorted[:, 1:]
        lm_f_targets, _ = pack_padded_sequence(lm_f_targets, lm_lengths, batch_first=True)

        # For the backward sequence, targets are <end> followed by all words except the last word
        # ...notsnud -> <end>, ...skcehc -> dunston, ...ni -> checks
        lm_b_targets = torch.cat(
            [torch.LongTensor([word_map['<end>']] * wmaps_sorted.size(0)).unsqueeze(1).to(device), wmaps_sorted], dim=1)
        lm_b_targets, _ = pack_padded_sequence(lm_b_targets, lm_lengths, batch_first=True)

        # Calculate loss
        ce_loss = lm_criterion(lm_f_scores, lm_f_targets) + lm_criterion(lm_b_scores, lm_b_targets)
        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)
        loss = ce_loss + vb_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))

        # Remove timesteps we won't predict at, and also <end> tags, because to predict them would be cheating
        decoded, _ = pack_padded_sequence(decoded, lm_lengths, batch_first=True)
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())
        tmaps_sorted, _ = pack_padded_sequence(tmaps_sorted, lm_lengths, batch_first=True)

        # F1
        f1 = f1_score(tmaps_sorted.to("cpu").numpy(), decoded.numpy(), average='macro')

        # Keep track of metrics
        ce_losses.update(ce_loss.item(), sum(lm_lengths))
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        batch_time.update(time.time() - start)
        f1s.update(f1, sum(lm_lengths))

        start = time.time()

        # Print training status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'CE Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                  'VB Loss {vb_loss.val:.4f} ({vb_loss.avg:.4f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})'.format(epoch, i, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time, ce_loss=ce_losses,
                                                          vb_loss=vb_losses, f1=f1s))


def validate(val_loader, model, crf_criterion, vb_decoder):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param crf_criterion: viterbi loss layer
    :param vb_decoder: viterbi decoder
    :return: validation F1 score
    """
    model.eval()

    batch_time = AverageMeter()
    vb_losses = AverageMeter()
    f1s = AverageMeter()

    start = time.time()

    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths) in enumerate(
            val_loader):

        max_word_len = max(wmap_lengths.tolist())
        max_char_len = max(cmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(device)
        cmaps_f = cmaps_f[:, :max_char_len].to(device)
        cmaps_b = cmaps_b[:, :max_char_len].to(device)
        cmarkers_f = cmarkers_f[:, :max_word_len].to(device)
        cmarkers_b = cmarkers_b[:, :max_word_len].to(device)
        tmaps = tmaps[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)
        cmap_lengths = cmap_lengths.to(device)

        # Forward prop.
        crf_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, __ = model(cmaps_f,
                                                                                   cmaps_b,
                                                                                   cmarkers_f,
                                                                                   cmarkers_b,
                                                                                   wmaps,
                                                                                   tmaps,
                                                                                   wmap_lengths,
                                                                                   cmap_lengths)

        # Viterbi / CRF layer loss
        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))

        # Remove timesteps we won't predict at, and also <end> tags, because to predict them would be cheating
        decoded, _ = pack_padded_sequence(decoded, (wmap_lengths_sorted - 1).tolist(), batch_first=True)
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())
        tmaps_sorted, _ = pack_padded_sequence(tmaps_sorted, (wmap_lengths_sorted - 1).tolist(), batch_first=True)

        # f1
        f1 = f1_score(tmaps_sorted.to("cpu").numpy(), decoded.numpy(), average='macro')

        # Keep track of metrics
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        f1s.update(f1, sum((wmap_lengths_sorted - 1).tolist()))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'VB Loss {vb_loss.val:.4f} ({vb_loss.avg:.4f})\t'
                  'F1 Score {f1.val:.3f} ({f1.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                  vb_loss=vb_losses, f1=f1s))

    print(
        '\n * LOSS - {vb_loss.avg:.3f}, F1 SCORE - {f1.avg:.3f}\n'.format(vb_loss=vb_losses,
                                                                          f1=f1s))

    return f1s.avg


if __name__ == '__main__':
    main()
