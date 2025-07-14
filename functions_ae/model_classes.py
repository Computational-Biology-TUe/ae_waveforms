import torch.nn as nn
from torch.nn.functional import relu


# ================================================= LSTM Autoencoders =================================================


class LSTMAutoencoder(nn.Module):
    def __init__(self, latent_size, dropout=0.2, hidden_sizes=(128, 64)):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_size=1, hidden_size=hidden_sizes[0], batch_first=True)
        self.encoder_lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
        self.encoder_lstm3 = nn.LSTM(input_size=hidden_sizes[1], hidden_size=latent_size, batch_first=True)
        # Decoder
        self.decoder_lstm1 = nn.LSTM(input_size=latent_size, hidden_size=hidden_sizes[1], batch_first=True)
        self.decoder_lstm2 = nn.LSTM(input_size=hidden_sizes[1], hidden_size=hidden_sizes[0], batch_first=True)
        self.decoder_lstm3 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=1, batch_first=True)
        # Output layer
        self.decoder_lin = nn.Linear(1, 1)

        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        x, _ = self.encoder_lstm1(x)
        x = self.dropout(x)
        x, _ = self.encoder_lstm2(x)
        x = self.dropout(x)
        x, (h_n3, _) = self.encoder_lstm3(x)
        return h_n3[-1]

    def decode(self, latent_space, seq_len):
        repeated_latent = latent_space.unsqueeze(1).repeat(1, seq_len, 1)
        x, _ = self.decoder_lstm1(repeated_latent)
        x = self.dropout(x)
        x, _ = self.decoder_lstm2(x)
        x = self.dropout(x)
        x, _ = self.decoder_lstm3(x)
        x = self.decoder_lin(x)
        return x

    def forward(self, x):
        latent_space = self.encode(x)
        return self.decode(latent_space, x.size(1))


# ============================================ Convolutional Autoencoders =============================================


class LeNet5AutoencoderAvgPool(nn.Module):
    def __init__(self, latent_size, dropout=0.2):
        super(LeNet5AutoencoderAvgPool, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.encoder_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.encoder_conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.encoder_conv_latent = nn.Conv1d(128, latent_size, kernel_size=1)  # Latent space (batch_size, latent_size, 1)

        self.pool = nn.AvgPool1d(2, stride=2)
        self.drop = nn.Dropout(dropout)

        # Decoder
        self.decoder_deconv_latent = nn.ConvTranspose1d(latent_size, 128, kernel_size=1)  # Expand to (batch_size, 128, 1)
        self.upsample1 = nn.Upsample(size=3, mode='linear')
        self.decoder_deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.Upsample(size=10, mode='linear')
        self.decoder_deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.Upsample(size=40, mode='linear')
        self.decoder_deconv3 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample4 = nn.Upsample(size=160, mode='linear')
        self.decoder_deconv4 = nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        # Encoder
        e1 = relu(self.encoder_conv1(x))  # (batch_size, 16, 160)
        p1 = self.pool(e1)  # (batch_size, 16, 80)
        e2 = relu(self.encoder_conv2(p1))  # (batch_size, 32, 40)
        if self.training:
            e2 = self.drop(e2)  # Dropout added after e2
        p2 = self.pool(e2)  # (batch_size, 32, 20)
        e3 = relu(self.encoder_conv3(p2))  # (batch_size, 64, 10)
        p3 = self.pool(e3)  # (batch_size, 64, 5)
        e4 = relu(self.encoder_conv4(p3))  # (batch_size, 128, 3)
        p4 = self.pool(e4)  # (batch_size, 128, 1)
        latent_space = self.encoder_conv_latent(p4)  # (batch_size, latent_size, 1)

        return latent_space

    def forward(self, x):
        latent_space = self.encode(x)  # Encode to latent space

        # Decoder
        l1 = relu(self.decoder_deconv_latent(latent_space))  # (batch_size, 128, 1)
        ul1 = self.upsample1(l1)  # (batch_size, 128, 3)
        d1 = relu(self.decoder_deconv1(ul1))  # (batch_size, 64, 5)
        u1 = self.upsample2(d1)  # (batch_size, 64, 10)
        d2 = relu(self.decoder_deconv2(u1))  # (batch_size, 32, 19)
        if self.training:
            d2 = self.drop(d2)  # Dropout added after d2
        u2 = self.upsample3(d2)  # (batch_size, 32, 40)
        d3 = relu(self.decoder_deconv3(u2))  # (batch_size, 16, 79)
        u3 = self.upsample4(d3)  # (batch_size, 16, 160)
        d4 = self.decoder_deconv4(u3)  # (batch_size, 1, 319)

        return d4


# ================================================== Model Classes ====================================================

model_classes = {
    'LSTMAutoencoder': LSTMAutoencoder,
    'LeNet5AutoencoderAvgPool': LeNet5AutoencoderAvgPool,
}
