from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_sequence_length):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_sequence_length = output_sequence_length

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -self.output_sequence_length :, :])
        return out
