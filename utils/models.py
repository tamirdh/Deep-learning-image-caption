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
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        # make the features tensor in the embed size length
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size) 
        self.relu = nn.ReLU()

    def forward(self, images):
        '''
        Input: image vector
        Output: features vector
        '''
        features = self.inception(images)
        # print("features size: ", features.size())
        output = self.relu(features)
        # print("output size: ", output.size())
        return output



class DecoderRNNV4(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3):
        super(DecoderRNNV4, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
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

    


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, train_CNN=False):
        global device
        device = get_device(1)
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size, train_CNN).to(device)
        self.decoderRNN = DecoderRNNV4(
            embed_size, hidden_size, vocab_size).to(device)

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


    def caption_images(self, features, vocab, max_len=77):
        '''
        Vec_len should be the same as is learning. 
        '''
        assert features.size(
            0) == 1, f"Caption features doesn't support batches got {features.shape}"
        # features: (B,F) -> (1,1,F)
        # w_embed: (1) -> (1,1,E)
        result_caption = []
        
        with torch.no_grad():
            x = self.encoderCNN(features).unsqueeze(0)
            states = None
            x = self.decoderRNN.fc_in(x)
            for _ in range(max_len):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.fc_out(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocab.itos[predicted.item()] == "<EOS>":
                    break

        return [vocab.itos[idx] for idx in result_caption]
    
    def train(self, mode=True):
        super().train(mode)
        if not self.encoderCNN.train_CNN:
            self.encoderCNN.eval()
        return self