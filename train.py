import numpy as np
import torch, json, tqdm, argparse, os
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from model import(
    Encoder,
    Decoder,
    EncoderDecoder,
    Attention
)

from utils import (
    get_device,
    build_tokenizer_table,
    build_output_tables,
    prefix_match,
    exact_match,
    encode_data
)


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    f = open('lang_to_sem_data.json')
    data = json.load(f) #keys = ['train', 'valid_seen']
    f.close()
    train_data=[]
    val_data=[]
    for i in data['train']:
        train_data.append(i)
    for i in data['valid_seen']:
        val_data.append(i)
    
    vocab_to_index, index_to_vocab, len_cutoff= build_tokenizer_table(train_data)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_data)
    x_train, y_train = encode_data(train_data, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
    x_val, y_val = encode_data(val_data, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
    # print(len_cutoff)

    # convert data from np to tensors
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    # define train and val dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size = args.batch_size)

    return train_loader, val_loader, len_cutoff, vocab_to_index, actions_to_index, targets_to_index


def setup_model(args, device, vocab_size, num_actions, num_targets):
    """
    return:
        - model: EncoderDecoder
    """
    # ===================================================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    # model_name: str, input_bos_token_id: int, input_eos_token_id: int, output_bos_token_id: int,
    # output_eos_token_id
    embedding_dim = 128
    hidden_dim = 128
    num_layers = 1
    dropout = 0

    return EncoderDecoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_actions, num_targets,
                               args.teacher_forcing, args.use_attention, device=device)                           



def setup_optimizer(args, model, device):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ===================================================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.CrossEntropyLoss(ignore_index=2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return criterion, optimizer


def get_loss_graph(args, action_losses, target_losses, output_fname, graph_title):
    if "Validation" in graph_title:
        x_axis_data = [i for i in range(0, args.num_epochs, args.val_every)] 
    else:
        x_axis_data = [i for i in range(args.num_epochs)]
    action_losses = [loss.cpu().detach().numpy() for loss in action_losses]
    target_losses = [loss.cpu().detach().numpy() for loss in target_losses]

    figure, ax = plt.subplots()
    ax.plot(x_axis_data, action_losses, label="Action")
    ax.plot(x_axis_data, target_losses, label="Target")
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_title(graph_title)
    path = os.path.join(args.outputs_dir, output_fname)

    figure.savefig(path)


def get_acc_graph(args, exact_matches, prefix_matches, output_fname, graph_title):
    if 'Validation' in graph_title:
        x_axis_data = [i for i in range(0, args.num_epochs, args.val_every)] 
    else:
        x_axis_data = [i for i in range(args.num_epochs)]

    figure, ax = plt.subplots()
    ax.plot(x_axis_data, exact_matches, label="Exact Match")
    ax.plot(x_axis_data, prefix_matches, label="Prefix Match")

    ax.legend()
    ax.set_xlabel("Num of epochs")
    ax.set_title(graph_title)
    path = os.path.join(args.outputs_dir, output_fname)
    
    figure.savefig(path)


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    """
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.

    --> Implemented in the EncoderDecoder model
    """

    epoch_loss_a, epoch_loss_t = 0.0, 0.0
    epoch_exact_match_acc_a, epoch_prefix_match_acc_a = 0.0, 0.0
    epoch_exact_match_acc_t, epoch_prefix_match_acc_t = 0.0, 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for inputs, labels in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        #get action anf target labels   
        action_labels = torch.zeros((len(labels), len(labels[0])))
        target_labels = torch.zeros((len(labels), len(labels[0])))
        for idx, label in enumerate(labels):
            action_labels[idx] = label[:, 0]
            target_labels[idx] = label[:, 1]
        action_labels, target_labels = action_labels.long().to(device), target_labels.long().to(device)

        #get seq lens of episode for inputs
        seq_lens = torch.zeros(len(inputs))
        for idx, input in enumerate(inputs):
            jdx = 0
            for jdx in range(len(input)):
                if input[jdx] == 0:
                    break
            seq_lens[idx] = jdx + 1
        
        #get seq lens of episode for labels
        labels_lens = torch.zeros(len(labels))
        for idx, label in enumerate(labels):
            label = torch.transpose(label, 0, 1)[0]
            jdx = 0
            for jdx in range(len(label)):
                if label[jdx] == 2:
                    break
            labels_lens[idx] = jdx + 1
        
        model = model.to(device)
        actions_pred, targets_pred, action_prob_dist, target_prob_dist = model(inputs, labels, seq_lens, teacher_force=training)

        #calculate loss
        action_loss = criterion(action_prob_dist, action_labels)
        target_loss = criterion(target_prob_dist, target_labels)
        loss = action_loss + target_loss
  

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #action scores
        action_exact_match_score = exact_match(actions_pred, action_labels, labels_lens)
        action_prefix_match_score = prefix_match(actions_pred, action_labels, labels_lens)

        #target scores
        target_exact_match_score = exact_match(targets_pred, target_labels, labels_lens)
        target_prefix_match_score = prefix_match(targets_pred, target_labels, labels_lens)

        #update values
        epoch_loss_a += action_loss
        epoch_exact_match_acc_a += action_exact_match_score
        epoch_prefix_match_acc_a += action_prefix_match_score
        epoch_loss_t += target_loss
        epoch_exact_match_acc_t += target_exact_match_score
        epoch_prefix_match_acc_t += target_prefix_match_score

    return (
        epoch_loss_a / len(loader),
        epoch_loss_t / len(loader),
        epoch_exact_match_acc_a / len(loader),
        epoch_prefix_match_acc_a / len(loader),
        epoch_exact_match_acc_t / len(loader),
        epoch_prefix_match_acc_t / len(loader)
    )


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_action_loss, val_target_loss, val_action_exact_match_acc, val_action_prefix_match_acc, val_target_exact_match_acc, val_target_prefix_match_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, val_action_exact_match_acc, val_action_prefix_match_acc, val_target_exact_match_acc, val_target_prefix_match_acc


def train(args, model, loaders, optimizer, criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation

    train_action_exact_match_accs, train_action_prefix_match_accs, train_action_losses = [],[],[]

    train_target_exact_match_accs, train_target_prefix_match_accs, train_target_losses = [],[],[]

    val_action_exact_match_accs, val_action_prefix_match_accs, val_action_losses = [],[],[]

    val_target_exact_match_accs, val_target_prefix_match_accs, val_target_losses = [],[],[]

    for epoch in tqdm.tqdm(range(args.num_epochs)):
        model = model.to(device)
        model.train()
        # returns loss for action and target prediction and accuracy
        train_action_loss, train_target_loss, train_action_exact_match_acc, train_action_prefix_match_acc, train_target_exact_match_acc, train_target_prefix_match_acc= train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        train_action_losses.append(train_action_loss)
        train_action_exact_match_accs.append(train_action_exact_match_acc)
        train_action_prefix_match_accs.append(train_action_prefix_match_acc)
        train_target_losses.append(train_target_loss)
        train_target_exact_match_accs.append(train_target_exact_match_acc)
        train_target_prefix_match_accs.append(train_target_prefix_match_acc)

        # some logging
        print(f"train_action_loss: {train_action_loss} \n train_action_exact_match_acc: {train_action_exact_match_acc} \n train_action_prefix_match_acc: {train_action_prefix_match_acc}")
        print('\n')
        print(f"train_target_loss: {train_target_loss} \n train_target_exact_match_acc: {train_target_exact_match_acc} \n train_target_prefix_match_acc: {train_target_prefix_match_acc}")
        


        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_exact_match_acc, val_action_prefix_match_acc, val_target_exact_match_acc, val_target_prefix_match_acc= validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )

            val_action_losses.append(val_action_loss)
            val_action_exact_match_accs.append(val_action_exact_match_acc)
            val_action_prefix_match_accs.append(val_action_prefix_match_acc)
            val_target_losses.append(val_target_loss)
            val_target_exact_match_accs.append(val_target_exact_match_acc)
            val_target_prefix_match_accs.append(val_target_prefix_match_acc)

            print(
                f"val_action_loss: {val_action_loss} \n val_action_exact_match_acc acc: {val_action_exact_match_acc} \n val_action_prefix_match_acc: {val_action_prefix_match_acc}")
            print('\n')
            print(
                f"val_target_loss: {val_target_loss} \n val__target_exact_match_acc acc: {val_target_exact_match_acc} \n val_target_prefix_match_acc: {val_target_prefix_match_acc}")

            # Save model
            ckpt_file = os.path.join(args.outputs_dir, args.model_fname)
            print("Saving model...", ckpt_file)
            torch.save(model, ckpt_file)

    # ===================================================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy, 3) validation loss, 4) validation accuracy
    # ===================================================== #

    get_loss_graph(args, train_action_losses, train_target_losses, "train_loss.png", "Train Loss")
    get_acc_graph(args, train_action_exact_match_accs, train_action_prefix_match_accs, "train_acc_action.png", "Train Accuracy(action)")
    get_acc_graph(args, train_target_exact_match_accs, train_target_prefix_match_accs, "train_acc_target.png", "Train Accuracy(target)")
    
    get_loss_graph(args, val_action_losses, val_target_losses, "val_loss.png", "Validation Loss")
    get_acc_graph(args, val_action_exact_match_accs, val_action_prefix_match_accs, "val_acc_action.png", "Validation Accuracy(action)")
    get_acc_graph(args, val_target_exact_match_accs, val_target_prefix_match_accs, "val_acc_target.png", "Validation Accuracy(target)")


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, len_cutoff, vocab_to_index, actions_to_index, targets_to_index = setup_dataloader(args)

    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device, len(vocab_to_index), len(actions_to_index), len(targets_to_index))
    print(model)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, model, device)

    if args.eval:
        _,_,_,_,_,_ = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--outputs_dir", type=str, help="where to save outputs"
    )
    parser.add_argument(
        "--model_fname", type=str, help="models filename"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", type=int, default=5, help="number of epochs between every eval loop"
    )
    parser.add_argument(
        "--teacher_forcing", action="store_true", help="use/dont use teacher forcing"
    )
    parser.add_argument(
        "--use_attention", action="store_true", help="use/dont use attention"
    )


    args = parser.parse_args()

    main(args)
