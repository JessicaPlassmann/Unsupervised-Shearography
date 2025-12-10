import torch


class AutoEncoderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(50 * 96, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.Dropout()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 50 * 96),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(-1, 50 * 96)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.reshape(-1, 1, 50, 96)
        return decoded
