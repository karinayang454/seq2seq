import re, os, torch
import numpy as np
from collections import Counter


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    n_steps = []
    for episode in train:
        padded_len = 2  # BOS/EOS
        n_step = 2 # BOS/EOS
        for inst, _ in episode:
            inst = preprocess_string(inst)
            n_step += 1
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
        padded_lens.append(padded_len)
        n_steps.append(n_step)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<BOS>"] = 1
    vocab_to_index["<EOS>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5)
        # max(n_steps)
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i+3 for i, a in enumerate(actions)}
    targets_to_index = {t: i+3 for i, t in enumerate(targets)}
    actions_to_index["<BOS>"], actions_to_index["<EOS>"] = 0, 1
    targets_to_index["<BOS>"], targets_to_index["<EOS>"] = 0, 1
    actions_to_index["<PAD>"] = 2
    targets_to_index["<PAD>"] = 2
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets


def prefix_match(predicted_labels, gt_labels, labels_len):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1
    bs = len(gt_labels)
    prefix_match = 0.0
    for i in range(bs):
        l = 0
        seq_len = int(labels_len[i].item())
        for l in range(seq_len):
            if (predicted_labels[i][l] != gt_labels[i][l]) or (predicted_labels[i][l] == 1 and gt_labels[i][l] == 1):
                break
        prefix_match += (l / seq_len)

    return prefix_match / bs


def exact_match(predicted_labels, gt_labels, labels_lens):
    bs = len(gt_labels)
    exact_match = 0.0
    for i in range(bs):
        x = True
        seq_len = int(labels_lens[i].item())
        for j in range(seq_len):
            if predicted_labels[i][j] != gt_labels[i][j]:
                x = False
                break
        exact_match += 1 if x else 0
    return exact_match / bs


def encode_data(data, vocab_to_index, seq_len, actions_to_index, targets_to_index):
    num_eps = len(data)
    x = np.zeros((num_eps, seq_len), dtype=np.int32)
    y = [] #action target pairs

    max_len_pair = 0
    for (idx, episode) in enumerate(data):
        x[idx][0] = vocab_to_index["<BOS>"]
        ep_y = [[actions_to_index["<BOS>"], targets_to_index["<BOS>"]]]
        
        jdx = 1
        for seq in episode:
            processed_instruction = preprocess_string(seq[0])
            action = seq[1][0]
            target = seq[1][1]

            for word in processed_instruction.split():
                if len(word) > 0:
                    x[idx][jdx] = vocab_to_index[word] if word in vocab_to_index else vocab_to_index[
                        "<unk>"]
                    jdx += 1
                    if jdx == seq_len - 1:
                        break
            ep_y.append([actions_to_index[action], targets_to_index[target]])
            if jdx == seq_len - 1:
                break
        # add stop to the seq and labels
        x[idx][jdx] = vocab_to_index["<EOS>"]
        ep_y.append([actions_to_index["<EOS>"], targets_to_index["<EOS>"]])

        # add to total y labels
        y.append(ep_y)
        max_len_pair = max(max_len_pair, len(ep_y))

    # pad to the max len pair
    for idx, episode_label in enumerate(y):
        y[idx].extend((max_len_pair - len(episode_label))*[[actions_to_index["<PAD>"], targets_to_index["<PAD>"]]])

    return np.array(x), np.array(y)


def get_episode_seq_lens(batch_input):
    batch_size = len(batch_input)
    batch_seq_lens = torch.zeros(batch_size)
    for idx, i_input in enumerate(batch_input):
        jdx = 0
        for jdx in range(len(i_input)):
            if i_input[jdx] == 0:
                break
        batch_seq_lens[idx] = jdx + 1
    return batch_seq_lens


def get_labels_seq_lens(batch_labels):
    batch_size = len(batch_labels)
    batch_seq_lens = torch.zeros(batch_size)
    for idx, i_label in enumerate(batch_labels):
        i_label = torch.transpose(i_label, 0, 1)[0]
        jdx = 0
        for jdx in range(len(i_label)):
            if i_label[jdx] == 2:
                break
        batch_seq_lens[idx] = jdx + 1
    return batch_seq_lens




