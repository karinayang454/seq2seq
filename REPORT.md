# Report

## To Run...

LSTM encoder-decoder model:
 - on CPU: `python train.py --in_data_fn=lang_to_sem_data.json --outputs_dir=outputs/experiments/s2s/ --model_fname=s2s_model.ckpt --batch_size=512 --num_epochs=10 --val_every=2 --force_cpu --teacher_forcing`

 - on GPU: `srun --gres=gpu:2080:1 --time 2:00:00 python train.py --in_data_fn=lang_to_sem_data.json --outputs_dir=outputs/experiments/s2s/ --model_fname=s2s_model.ckpt --batch_size=512 --num_epochs=10 --val_every=2 --teacher_forcing > outputs.txt` OR `sh run.sh`

LSTM encoder-decoder model with attention:
 - on CPU: `python train.py --in_data_fn=lang_to_sem_data.json --outputs_dir=outputs/experiments/s2s_attention/ --model_fname=s2s_attention_model.ckpt --batch_size=128 --num_epochs=8 --val_every=2 --force_cpu --teacher_forcing --use_attention`

 - on GPU: `srun --gres=gpu:2080:1 --time 2:00:00 python train.py --in_data_fn=lang_to_sem_data.json --outputs_dir=outputs/experiments/s2s_attention/ --model_fname=s2s_attention_model.ckpt --batch_size=512 --num_epochs=10 --val_every=2 --teacher_forcing --use_attention > attention_outputs.txt` OR `sh run_attention.sh`

## Implementation

### Hyperparameters
I used batch_size = 512, n_epochs = 10, with validation every 2 epochs. I applied the Adam optimizer with a learning rate of 0.001, and cross-entropy loss to classify multiple classes (for both actions and targets). High batch size was chosen to increase computation speed, and other hyperparameters were based on default values.
  - Model input: all instructions per episode were concatenated together as one data sample
  - Model output: a sequence of actions and targets for each episode


I implemented an encoder-decoder model with optional attention mechanism. 

### Encoder
The encoder reads the sequence inputs in an episode, and produces a sequence of hidden states for each source word. I first embed the input in a word embedding with vocab size 1000, and embedding dimension 128. Then, I feed this into 1 LSTM layer with a hidden dimension of 128. These numbers were chosen rather arbitrarily, but mostly to minimize computation time during training. I also used `pack_padded_sequence` and `pad_packed_sequence` for the LSTM to ignore the paddings during training. The encoder then returns the final output, hidden states, and cell states of the LSTM.

### Decoder 
The decoder iteratively generates the next token given the context vector from the encoder and ground truth labels using teacher forcing. The initial state of the decoder is first set to the final output state of the encoder. Each LSTM cell uses the previous hidden state, and the embeddings of concatenated gold actions and targets as input. The embedding dimesnion here is also 128, with 1 LSTM layer using a hidden dimension of 256. To predict the final action and targets, 2 linear layers with output size 11 and 83 were applied to the output of each LSTM cell, respectively, to correspond to the number of action and target classes. 

### Attention
The attention mechanism is a global, soft attention, attends to a word-level, and was implemented via a linear layer. For all decoder timesteps, I duplicate and concatenate the current decoder hidden state with each individual hidden state of the encoder, where a linear layer + softmax then calculates the weights of each encoder hidden representation.


## Model Performance

### Evaluation Metrics
I applied 2 different metrics:
- Exact Match: measures frequency of gold sequences and predicted sequences being the exact same
- Prefix match: measures proportion of matching action/target labels there are between predicted sequences and gold sequences

### Encoder-Decoder Performance

- Train action loss: 0.2895942
- Train action exact match: 0.0001132
- Train action prefix match: 0.7663735

- Train target loss: 0.6199424
- Train target exact match: 2.8306159e-05
- Train target prefix match: 0.6563111

- Val action loss: 0.7746886
- Val action exact match: 0.0
- Val action prefix match: 0.7336871

- Val target loss: 1.4130136
- Val target exact match: 0.0
- Val target prefix match: 0.5009542

### Encoder-Decoder with Attention Performance

- Train action loss: 1.5295308
- Train action exact match: 0.0
- Train action prefix match: 0.2172919

- Train target loss: 1.9198582
- Train target exact match: 0.0
- Train target prefix match: 0.1325686

- Val action loss: 1.5480083
- Val action exact match: 0.0
- Val action prefix match: 0.2122143

- Val target loss: 2.0890040
- Val target exact match: 0.0
- Val target prefix match: 0.1281798

For both models, action prediction performed better than target prediction, which is expected since there are less classes in action (11 vs. 83). Exact match for both models was also near 0, as expected, since the output sequences are quite long. 

The encoder decoder with the additional attention mechanism performed significantly worse compared to the model without attention. It was likely difficult for the model to effectively learn which word to attend to during training, so the noisy information from attention would have caused the poor performance. Perhaps if I used instruction level attention instead, or if I simply trained for a longer period of time, the attention-based model would perform better. 

Training and Validation graphs can be found in `outputs/experiments/s2s` and `outputs/experiments/s2s_attention`.