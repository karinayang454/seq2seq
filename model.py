import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, device):
        super(Encoder, self).__init__()

        self.device = device
        self.emb_layer = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)
        self.vocab_size = vocab_size
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True)

    def forward(self, episodes, seq_lens):
        # episodes: bs, seq length
        # seq_lens: len = bs
        # print(episodes.size(), '---------')
        emb_out = self.emb_layer(episodes) # bs, seq len, emb_dim
        emb_out = pack_padded_sequence(emb_out, lengths=seq_lens, batch_first=True, enforce_sorted = False).to(self.device)

        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # output: bs, seq len, hidden size
        # hn: 1, bs, hidden size
        # cn: 1, bs, hiddne size
        self.LSTM = self.LSTM.to(self.device)
        output, (hn, cn) = self.LSTM(emb_out)

        padded_output, _ = pad_packed_sequence(output, batch_first=True)

        return padded_output, hn, cn


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    input: concatenated one hot encoding of gold a and o, prev hidden state h_{t-1}
    output: 2 FC layers one of size 8 and one of size 80 --> softmax and loss, new hiden state h_{t}

    during training, we use gold a and o, during inference, you must use previously predicted

    input to LSTM: B x N x L = 1 x h with batch first = true
    hidden: 1 x B x h

    attention = linear layer + softmax
    """

    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, num_actions, num__targets,
                 use_attn, device):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.use_attn = use_attn
        self.device = device

        #create embedding layers
        self.emb_layer_action = nn.Embedding(num_embeddings=num_actions, embedding_dim=embedding_dim)
        self.emb_layer_target = nn.Embedding(num_embeddings=num__targets, embedding_dim=embedding_dim)

        #LSTM
        self.LSTM = nn.LSTM(input_size=embedding_dim*2, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout)

        # linear layers for each action target output
        self.fc_action = nn.Linear(hidden_dim, num_actions)
        self.fc_target = nn.Linear(hidden_dim, num__targets)

        if use_attn:
            self.attention = Attention(hidden_dim, device)

    def forward(self, label, hn, cn, enc_hidden_outputs):

        # create and concat action and target embs
        action_embedding_out = self.emb_layer_action(label[0])
        target_embedding_out = self.emb_layer_target(label[1])
        embedding_out = torch.concat((action_embedding_out, target_embedding_out), dim=1).unsqueeze(0)

        # get LSTM output, hn and cn
        lstm_out, (new_hn, new_cn) = self.LSTM(embedding_out, (hn, cn))
        if self.use_attn:
            lstm_out = self.attention(new_hn[self.num_layers - 1], enc_hidden_outputs).unsqueeze(0)

        action_output = self.fc_action(lstm_out).squeeze(0)
        target_output = self.fc_target(lstm_out).squeeze(0)

        return action_output, target_output, new_hn, new_cn


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_actions, num_targets,
                 teacher_force = True, use_attn = False, device="cpu"):
        super(EncoderDecoder, self).__init__()

        #init vars
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers       
        self.num_actions = num_actions
        self.num_targets = num_targets
        self.teacher_force = teacher_force
        self.use_attn = use_attn
        self.device = device

        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, device)
        
        self.decoder = Decoder(embedding_dim, hidden_dim, num_layers, dropout, num_actions, num_targets, self.use_attn, device)


    def forward(self, episodes, labels, seq_lens, teacher_force=True):

        bs, num_instructions = len(labels), len(labels[0])
        labels = torch.transpose(labels, 0, 1)


        encoder_lstm_out, hn, cn = self.encoder(episodes, seq_lens)

        # init vars to store predicted distribution across actions and targets
        action_distribution = torch.zeros((num_instructions, bs, self.num_actions), device=self.device)
        target_distribution = torch.zeros((num_instructions, bs, self.num_targets), device=self.device)

        # init action & target for each sample in the batch
        actions_pred = torch.zeros((num_instructions, bs))
        targets_pred = torch.zeros((num_instructions, bs))

        # begin w <BOS> tokens
        pred_action_target = torch.zeros((2, bs), dtype=torch.long, device=self.device)

        for i in range(1, num_instructions):
            action_out, target_out, hn, cn = self.decoder(pred_action_target, hn, cn, encoder_lstm_out)
            
            predicted_action = torch.argmax(action_out, dim=1)
            predicted_target = torch.argmax(target_out, dim=1)
            # concatenate the preds togeher
            pred_action_target = torch.concat((predicted_action, predicted_target)).reshape(2, bs)

            # update results
            action_distribution[i] = action_out
            target_distribution[i] = target_out
            actions_pred[i] = predicted_action
            targets_pred[i] = predicted_target

            # Use gold_labels for teacher_force
            if teacher_force:
                pred_action_target = torch.transpose(labels[i], 0, 1) 

        # reshape
        actions_pred = torch.transpose(actions_pred, 0, 1)
        targets_pred = torch.transpose(targets_pred, 0, 1)
        action_dist = torch.transpose(action_distribution, 0, 1)
        action_dist = torch.transpose(action_dist, 1, 2)
        target_dist = torch.transpose(target_distribution, 0, 1)
        target_dist = torch.transpose(target_dist, 1, 2)

        return actions_pred, targets_pred, action_dist, target_dist


class Attention(nn.Module):
    def __init__(self, hidden_dim, device):
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.attn_score = nn.Linear(in_features=2 * hidden_dim, out_features=1)
        self.device = device

    def forward(self, decoder_hidden, enc_hidden_outputs):
        # decoder hidden states: bs, hidden dim
        # encoder hidden state out: bs, sequence length, hidden dim

        bs, seq_len = int(enc_hidden_outputs.shape[0]), int(enc_hidden_outputs.shape[1])

        # concatenate every encoder hs w decoder hs
        rep_decoder_hidden = torch.zeros((bs, seq_len, self.hidden_dim), device = self.device)
        for i in range(len(decoder_hidden)):
            rep_decoder_hidden[i] = decoder_hidden[i].repeat(1, seq_len, 1)

        concatenated_hidden = torch.cat((rep_decoder_hidden, enc_hidden_outputs), dim=2)

        # get attn scores
        attn_scores = self.attn_score(concatenated_hidden)
        weights = F.softmax(attn_scores, dim=1)
        return torch.bmm(enc_hidden_outputs.transpose(1, 2), weights).squeeze(2)
