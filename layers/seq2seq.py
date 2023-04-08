import torch
import torch.nn as nn
import numpy as np


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        self.gru = nn.GRU(input_size, hidden_size*30, num_layers, batch_first=True)
        
    def forward(self, inputs):
        output, hidden_state = self.gru(inputs)
        return output, hidden_state
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.5, isCuda=False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        self.gru = nn.GRU(hidden_size, output_size*30, num_layers, batch_first=True)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(output_size*30, output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, encoded_input, hidden):
        decoded_output, hidden = self.gru(encoded_input, hidden)
        
        decoded_output = self.dropout(decoded_output)
        
        decoded_output = self.linear(decoded_output)
        
        return decoded_output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, isCuda=False):
        super(Seq2Seq, self).__init__()
        self.isCuda = isCuda
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, hidden_size, num_layers, dropout, isCuda)
    
    def forward(self, in_data, last_location, pred_length, teacher_forcing_ratio=0, teacher_location=None):
        batch_size = in_data.shape[0]
        out_dim = self.decoder.output_size
        self.pred_length = pred_length
        
        outputs = torch.zeros(batch_size, self.pred_length, out_dim)
        if self.isCuda:
            outputs = outputs.cuda()
        
        encoded_output, hidden = self.encoder(in_data)
        decoder_input = last_location
        for t in range(self.pred_length):
            now_out, hidden = self.decoder(decoder_input, hidden)
            now_out += decoder_input
            outputs[:, t:t+1] = now_out
            teacher_force = np.random.random() < teacher_forcing_ratio
            decoder_input = (teacher_location[:, t:t+1] if type(teacher_force) is not type(None) and teacher_force else now_out)
        
        return outputs
            