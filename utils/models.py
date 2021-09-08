import torch
import torch.nn as nn
import torchvision.models as models
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        self.cnn = models.resnet50(
            pretrained=not self.train_CNN)
        self.cnn.fc = nn.Linear(
            self.cnn.fc.in_features, output_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, images):
        '''
        Input: image vector
        Output: features vector
        '''
        features = self.cnn(images)
        return self.softmax(features)


class EncoderCNNV2(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        # CNN of pretrained ResNet-152
        super(EncoderCNNV2, self).__init__()
        resnet = models.resnet152(pretrained=True)
        # delete the last fc layer.
        modules = list(resnet.children())[:-1]      
        self.cnn = nn.Sequential(*modules)
        self.fc_out = nn.Linear(resnet.fc.in_features, embed_size)
        self.norm = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        # Extract feature vectors from input images.
        with torch.no_grad():
            features = self.cnn(images)
        features = features.reshape(features.size(0), -1)
        features = self.norm(self.fc_out(features))
        return features



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, n_features = 0):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm_cell = nn.LSTMCell(
            input_size=embed_size, hidden_size=hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions, len):
        '''
        features: Tensor, (B, S_in)
        captions: Tensor, (B, S_cap)
        '''
        # batch size
        batch_size = features.size(0)
        show=False
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
        self.num_layers = 1
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size+n_features,
                            hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.2)
        self.fc_out = nn.Linear(in_features=hidden_size,
                                out_features=vocab_size)

    def forward(self, features, captions, len):
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


class DecoderRNNV3(DecoderRNNV2):
    def __init__(self, embed_size, hidden_size, vocab_size, n_features):
        super().__init__(embed_size, hidden_size, vocab_size, n_features)

    def forward(self, features, captions, cap_lengths):
        # cap_lengths - list of the real length of each caption before padding
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
        # create packedSequence that is better for LSTM
        packed = pack_padded_sequence(
            combined, cap_lengths, batch_first=True, enforce_sorted=False)
        # run through the LSTM network and get output of shape (B, L, H)
        lstm_out, _ = self.lstm(packed)
        # unpack so we can use Linear function (works on Tensor not packSeq)
        output_padded, output_lengths = pad_packed_sequence(
            lstm_out, batch_first=True)

        return self.fc_out(output_padded)

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


class DecoderRNNEGreed(DecoderRNNV2):
    def __init__(self, embed_size, hidden_size, vocab_size, n_features):
        super().__init__(embed_size, hidden_size, vocab_size, n_features)
        self.eps_greedy = False
        self.counter = 0
        self.greed_selector = 1

    def forward(self, features, captions, cap_lengths):
        '''
        Uses a combination of the image and caption vector in the lstm
        to predict each word in the embedding layer
        '''
        # validate inputs are the same batch size
        assert features.size(0) == captions.size(0)
        batch_size = features.size(0)
        features = torch.unsqueeze(features, 1)

        if self.counter > 10000:
            self.eps_greedy = True
        # (h_0, c_0) will be initialized to zeros by default
        if not self.eps_greedy or self.use_caption_eps_greedy():
            # Advance counter towards eps greedy policy
            self.counter += 1
            # embed captions, shape (B, L, E)
            captions_embed = self.embed(captions)
            # features, shape (B, F)
            # features transform shape to (B, L, F)
            # (1,1,2048) -> (1,77, 2048)
            features = features.repeat((1, captions_embed.size(1), 1))
            # combine features + captions to shape (B, L, E+F) (1,77,2048) -> (1,77,2448)
            combined = torch.cat((features, captions_embed), dim=2)
            packed = pack_padded_sequence(combined, cap_lengths, batch_first=True, enforce_sorted=False)
            # run through the LSTM network and get output of shape (B, L, H)
            lstm_out, _ = self.lstm(packed)
            output_padded, _ = pad_packed_sequence(lstm_out, batch_first=True)
            return self.fc_out(output_padded)
        else:
            # features: (B,F) -> (B,1,F)
            # w_embed: (1) -> (B,1,E)
            # W0 is <SOS>
            w0 = torch.tensor([1]).to(device)
            w0 = w0.repeat((batch_size, 1))
            w_embed = self.embed(w0)
            hi = None
            ci = None
            output = list()
            for i in range(captions.size(1)):
                combined = torch.cat((features, w_embed), dim=2)
                if i == 0:
                    lstm_out, (hi, ci) = self.lstm(combined)
                else:
                    lstm_out, (hi, ci) = self.lstm(combined, (hi, ci))
                res = self.fc_out(lstm_out)
                output.append(res)
                next_w = torch.argmax(res, dim=2)
                # lstm_out: (1,1,H)
                # hi, ci: (num_layers, 1, H)
                # next_w: (1,1,vocab_size)
                w_embed = self.embed(next_w)
            return torch.cat(output, dim=1)

    def use_caption_eps_greedy(self) -> bool:
        n = random.uniform(0, 1)
        eps = 1/self.greed_selector
        self.greed_selector += 1
        if eps >= n:
            return True
        else:
            return False

    def caption_features(self, features, vocab, vec_len):
        '''
        Vec_len should be the same as is learning. 
        '''
        assert features.size(
            0) == 1, "Caption features doesn't support batches"
        # features: (B,F) -> (1,1,F)
        # w_embed: (1) -> (1,1,E)
        w0 = torch.tensor(vocab.stoi["<SOS>"]).to(device)
        w0 = w0.repeat((1, 1))
        w_embed = self.embed(w0)
        features = torch.unsqueeze(features, 1)
        hi = None
        ci = None
        output = list()
        for i in range(vec_len):
            combined = torch.cat((features, w_embed), dim=2)
            if i == 0:
                lstm_out, (hi, ci) = self.lstm(combined)
            else:
                lstm_out, (hi, ci) = self.lstm(combined, (hi, ci))
            res = self.fc_out(lstm_out)
            next_w = torch.argmax(res, dim=2)
            output.append(vocab.itos[next_w.item()])
            # lstm_out: (1,1,F)
            # hi, ci: (num_layers, 1, F)
            # next_w: (1,1,vocab_size)
            if next_w.item() == vocab.stoi["<EOS>"]:
                break
            w_embed = self.embed(next_w)
        return output

class DecoderRNNV4(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, not_used):
        super(DecoderRNNV4, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = 3
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, features, captions, cap_lengths):
        # cap_lengths - list of the real length of each caption before padding
        assert features.size(0) == captions.size(0)

        # embed captions, shape (B, L, E)
        captions_embed = self.embed(captions)
        # features, shape (B, E)
        # features transform shape to (B, L, E)
        features = torch.unsqueeze(features, dim=1)  # (1,256) -> (1,1,256)
        
        # (1,1,256) -> (1,77, 256)
        # features = features.repeat((1, captions_embed.size(1), 1))
        # print("features size 2:", features.size())
        # combine features + captions to shape (B, 1+L, E) (1,1,256) -> (1,14,256)
        combined = torch.cat((features, captions_embed), dim=1)
        
        # create packedSequence that is better for LSTM
        packed = pack_padded_sequence(
            combined, cap_lengths, batch_first=True, enforce_sorted=False)
        # run through the LSTM network and get output of shape (B, L, H)
        lstm_out, _ = self.lstm(packed)
        # unpack so we can use Linear function (works on Tensor not packSeq)
        output_padded, output_lengths = pad_packed_sequence(
            lstm_out, batch_first=True)

        return self.fc_out(output_padded)

    def caption_features(self, features, vocabulary, max_len=77):
        """Generate captions for given image features using greedy search."""
        states = None
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.fc_out(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
            

        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = sampled_ids[0].cpu().numpy()               # (1, max_len) -> (max_len)
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocabulary.itos[word_id]
            sampled_caption.append(word)
            if word == "<EOS>":
                break
        return sampled_caption




class DecoderRNNV5(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, not_used):
        super(DecoderRNNV5, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = 1
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(in_features=2*hidden_size, out_features=vocab_size)
        self.fc_in = nn.Linear(in_features=15*embed_size, out_features=embed_size)

    def forward(self, features, captions, cap_lengths):
        # cap_lengths - list of the real length of each caption before padding
        assert features.size(0) == captions.size(0)
        features = self.fc_in(features)
        # embed captions, shape (B, L, E)
        captions_embed = self.embed(captions)
        # features, shape (B, E)
        # features transform shape to (B, L, E)
        features = torch.unsqueeze(features, dim=1)  # (1,256) -> (1,1,256)
        
        # (1,1,256) -> (1,77, 256)
        # features = features.repeat((1, captions_embed.size(1), 1))
        # print("features size 2:", features.size())
        # combine features + captions to shape (B, 1+L, E) (1,1,256) -> (1,14,256)
        combined = torch.cat((features, captions_embed), dim=1)
        
        # create packedSequence that is better for LSTM
        packed = pack_padded_sequence(
            combined, cap_lengths, batch_first=True, enforce_sorted=False)
        # run through the LSTM network and get output of shape (B, L, H)
        lstm_out, _ = self.lstm(packed)
        # unpack so we can use Linear function (works on Tensor not packSeq)
        output_padded, output_lengths = pad_packed_sequence(
            lstm_out, batch_first=True)

        return self.fc_out(output_padded)

    def caption_features(self, features, vocab, max_len=77):
        '''
        Vec_len should be the same as is learning. 
        '''
        assert features.size(
            0) == 1, f"Caption features doesn't support batches got {features.shape}"
        # features: (B,F) -> (1,1,F)
        # w_embed: (1) -> (1,1,E)
        result_caption = []
        x = features.unsqueeze(0)
        start = vocab.stoi["<SOS>"]
        start = torch.tensor(start).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            states = None
            x = self.decoderRNN.fc_in(x)
            x = torch.cat((x, self.decoderRNN.embed(start)), dim=1)
            for i in range(max_len):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.fc_out(hiddens.squeeze(0))
                predicted = output.argmax(1)
                #if i ==0:
                #    result_caption.extend([i.item() for i in predicted])
                #else:
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if any([vocab.itos[i] == "<EOS>" for i in result_caption]):
                    break

        return [vocab.itos[idx] for idx in result_caption]
    

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, n_features, train_CNN=False, start = True):
        global device
        device = get_device(1)
        super(CNNtoRNN, self).__init__()
        if start:
            self.encoderCNN = EncoderCNNV2(embed_size, train_CNN).to(device)
            self.decoderRNN = DecoderRNNV4(embed_size, hidden_size, vocab_size, n_features).to(device)
        else: 
            self.encoderCNN = None
            self.decoderRNN = None

    def forward(self, images, captions, length):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions, length)
        return outputs

    # def caption_images(self, image, vocab, max_len=50):
    #     # Inference part
    #     # Given the image features generate the captions
    #     features = self.encoderCNN(image)
    #     decoded = self.decoderRNN.caption_features(features, vocab, max_len)
    #     return decoded


    def caption_image(self, image, vocab, max_len):
        with torch.no_grad():
            features = self.encoderCNN(image)
        output = self.decoderRNN.caption_features(features, vocab, max_len)
        return output

    '''def train(self, mode=True):
        super().train(mode)
        if not self.encoderCNN.train_CNN:
            self.encoderCNN.eval()
        return self
    '''