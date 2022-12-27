
import torch.nn as nn
from torchvision import models
import torch

class LSTMNetwork(nn.Module):
    def __init__(self, hidden_dim, embed_dim, vocab_size, num_layers, model_type='LSTM'):
        super().__init__()
        self.model_type = model_type
        # Encoder
        self.pretrained_resnet = self.load_resnet(embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Decoder
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, nonlinearity='relu', batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
    def load_resnet(self, embed_dim):
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        feature_num = resnet.fc.in_features
        resnet.fc = nn.Linear(feature_num,embed_dim)    
        return resnet
    def forward(self, images, captions, lengths):
        f1 = self.pretrained_resnet(images)
        f2 = self.bn(f1)
        embeddings = self.embedding(captions)
        f_with_embed = torch.cat((f2.unsqueeze(1),embeddings),dim=1)
        packed_f_with_embed = nn.utils.rnn.pack_padded_sequence(f_with_embed, lengths, batch_first=True)
        if self.model_type.upper() == 'LSTM':
            packed_f3, temp = self.lstm(packed_f_with_embed)
        else:
            packed_f3, temp = self.rnn(packed_f_with_embed)
        outputs = self.linear(packed_f3.data)
        return outputs
    def generate_caption(self, image, vocabulary, is_stochastic=True, max_length=50, temperature=0.8, device=None):
        # We cannot use teacher forcing in generating captions, therefore we cannot use standard forward pass here.
        # We need to feed only the images to the network, without the captions and let the network run on its own.
        result_caption = []
        with torch.no_grad():
            features = self.pretrained_resnet(image.unsqueeze(0).to(device))
            features = self.bn(features)
            input = features.unsqueeze(0)
            states = None
            for i in range(max_length):

                if self.model_type.upper() == 'LSTM':
                    hiddens, states = self.lstm(input, states)
                else:
                    hiddens, states = self.rnn(input, states)
                hiddens = hiddens.squeeze(1)
                outputs = self.linear(hiddens)
                if is_stochastic:
                    probs = nn.functional.softmax(outputs/temperature, dim=1)
                    predictions = torch.multinomial(probs, 1)
                else:
                    predictions = outputs.argmax(1)

                result_caption.append(predictions.item())

                #Add embedding
                input = self.embedding(predictions)

                if vocabulary.idx2word[predictions.item()] == "<start>":
                    continue
                if vocabulary.idx2word[predictions.item()] == "<EOS>" or vocabulary.idx2word[predictions.item()] == "<end>":
                    break

        return [vocabulary.idx2word[idx] for idx in result_caption]