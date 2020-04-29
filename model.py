import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = 0.4)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax()
        self.hidden_size = hidden_size
    def forward(self, features, captions):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        batch_size = features.size(0)
        self.hidden = ((torch.zeros((1,batch_size, self.hidden_size))).to(device),
                       (torch.zeros((1,batch_size, self.hidden_size))).to(device))
        
        word_embed = self.embedding(captions[:,:-1])
        vis_text = torch.cat((features.unsqueeze(1),word_embed), dim = 1)

        output, self.hidden = self.lstm(vis_text, self.hidden)
        
        linearized = self.linear(output)
        return linearized

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        samples = []
        #start_capt = torch.cuda.LongTensor([0])
        #init_embedding = self.embedding(start_capt)
        if states is None:
            states = (torch.randn(1, 1, self.hidden_size).to(inputs.device), torch.randn(1, 1,
                                                                                         self.hidden_size).to(inputs.device))
        out, states = self.lstm(inputs, states)
        
        linear1 = self.linear(out)
        probab = linear1
        idx = np.argmax(probab.cpu().detach().numpy().flatten())
        start_capt = torch.cuda.LongTensor([idx])
        embedding = self.embedding(start_capt)
        embedding = embedding.view(1,embedding.size(0),-1)
        for i in range(max_len):
            output, states = self.lstm(embedding, states)
         
            linear = self.linear(output)
            #probab = self.softmax(linear)
            probab = linear
            idx = np.argmax(probab.cpu().detach().numpy().flatten())
            if int(idx) == 1:
                break
            samples.append(int(idx))
            start_capt = torch.cuda.LongTensor([idx])
            embedding = self.embedding(start_capt)
            embedding = embedding.view(1,embedding.size(0),-1)
        
        return samples