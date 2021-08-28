import torch
import torch.nn as nn
import torchvision.models as models

device = None


def get_device(gpus=1):
    if gpus == 1:
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if torch.cuda.is_available():
            return f"cuda:{gpus-1}"
        else:
            return "cpu"


class EncoderCNN(nn.Module):
    def __init__(self, output_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained= not self.train_CNN, aux_logits=False)
        self.inception.fc = nn.Linear(
            self.inception.fc.in_features, output_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        '''
        Input: image vector
        Output: features vector
        '''
        features = self.inception(images)
        output = self.relu(features)
        return output


class DecoderRNNV2(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, n_features):
        '''
        L- caption vec length
        B- batch size
        E- embed size
        H- hidden size
        F- number of features from CNN
        '''
        super(DecoderRNNV2, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = 3
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size+n_features,
                            hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_out = nn.Linear(in_features=hidden_size,
                                out_features=vocab_size)

    def forward(self, features, captions):
        '''
        Uses a combination of the image and caption vector in the lstm
        to predict each word in the embedding layer
        '''
        # validate inputs are the same batch size
        assert features.size(0) == captions.size(0)
        # (h_0, c_0) will be initialized to zeros by default
        # embed captions, shape (B, L, E)
        captions_embed = self.embed(captions)
        # features, shape (B, F)
        # features transform shape to (B, L, F)
        features = torch.unsqueeze(features, dim=1)  # (1,2048) -> (1,1,2048)
        # (1,1,2048) -> (1,77, 2048)
        features = features.repeat((1, captions_embed.size(1), 1))
        # combine features + captions to shape (B, L, E+F) (1,77,2048) -> (1,77,2448)
        combined = torch.cat((features, captions_embed), dim=2)
        # run through the LSTM network and get output of shape (B, L, H)
        lstm_out, _ = self.lstm(combined)
        return self.fc_out(lstm_out)

    def caption_features(self, features, vocab, vec_len):
        '''
        Vec_len should be the same as is learning. 
        '''
        assert features.size(
            0) == 1, "Caption features doesn't support batches"
        # features: (B,F) -> (1,1,F)
        # w_embed: (1) -> (1,1,E)
        w0 = torch.tensor(vocab.stoi["<SOS>"]).to(device)
        w0 = torch.unsqueeze(w0, 0)
        w0 = torch.unsqueeze(w0, 0)
        w_embed = self.embed(w0)
        features = torch.unsqueeze(features, 1)
        hi = torch.zeros((self.num_layers, 1, self.hidden_size)).to(device)
        ci = torch.zeros((self.num_layers, 1, self.hidden_size)).to(device)
        output = ["<SOS>"]
        for i in range(vec_len):
            combined = torch.cat((features, w_embed), dim=2)
            if i == 0:
                lstm_out, (hi, ci) = self.lstm(combined)
            else:
                lstm_out, (hi, ci) = self.lstm(combined, (hi, ci))
            next_w = torch.argmax(self.fc_out(lstm_out), dim=2)
            output.append(vocab.itos[next_w.item()])
            # lstm_out: (1,1,F)
            # hi, ci: (num_layers, 1, F)
            # next_w: (1,1,vocab_size)
            w_embed = self.embed(next_w)
        return output


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm_cell = nn.LSTMCell(
            input_size=embed_size, hidden_size=hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions, show=False):
        '''
        features: Tensor, (B, S_in)
        captions: Tensor, (B, S_cap)
        '''
        # batch size
        batch_size = features.size(0)

        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        hidden_state, cell_state = self.lstm_cell(
            features, (hidden_state, cell_state))
        # define the output tensor placeholder
        outputs = torch.empty(
            (batch_size, captions.size(1), self.vocab_size)).to(device)

        # embed the captions
        captions_embed = self.embed(captions)
        # tensor of shape (B, LEN, EMBED SIZE)
        # LEN- vectors length (longest caption+2)

        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the feature vector
            # if t == 0:
            #     hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))

            # # for the 2nd+ time step, using teacher forcer
            # else:
            hidden_state, cell_state = self.lstm_cell(
                captions_embed[:, t, :], (hidden_state, cell_state))
            # output of the attention mechanism
            out = self.fc_out(self.dropout(hidden_state))
            # build the output tensor
            outputs[:, t, :] = out
        if show:
            # print(f"Captions:{captions}")
            #print(f"outputs shape:{outputs.shape}")
            pass
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, features, embed_size, hidden_size, vocab_size, train_CNN=False):
        global device
        device = get_device(1)
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(features, train_CNN).to(device)
        self.decoderRNN = DecoderRNNV2(
            embed_size, hidden_size, vocab_size, features).to(device)

    def forward(self, images, captions, show=False):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_images(self, image, vocab, max_len=50):
        # Inference part
        # Given the image features generate the captions
        features = self.encoderCNN(image)
        decoded = self.decoderRNN.caption_features(features, vocab, max_len)
        return decoded
