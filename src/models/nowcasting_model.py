import torch
import torch.nn as nn


class ReliabilityAwareFusion(nn.Module):

    def __init__(self, feature_channels=12):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(feature_channels * 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, availability):

        x = torch.nan_to_num(x, nan=0.0)

        fused_input = torch.cat([x, availability], dim=1)

        return self.encoder(fused_input)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, hidden_state):

        h_prev, c_prev = hidden_state

        combined = torch.cat([x, h_prev], dim=1)

        gates = self.gates(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c

    def init_hidden(self, batch_size, height, width, device):

        h = torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=device,
        )

        c = torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=device,
        )

        return h, c


class ThunderstormNowcaster(nn.Module):

    def __init__(
        self,
        input_channels=12,
        hidden_channels=64,
        horizons=4,
    ):
        super().__init__()

        self.fusion = ReliabilityAwareFusion(input_channels)

        self.convlstm = ConvLSTMCell(
            input_channels=64,
            hidden_channels=hidden_channels,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, horizons, kernel_size=1),

            nn.Sigmoid(),
        )

    def forward(self, x, availability):

        batch_size, time_steps, _, height, width = x.shape

        h, c = self.convlstm.init_hidden(
            batch_size,
            height,
            width,
            x.device,
        )

        for t in range(time_steps):

            fused = self.fusion(
                x[:, t],
                availability[:, t],
            )

            h, c = self.convlstm(
                fused,
                (h, c),
            )

        output = self.decoder(h)

        return output


def test_model():

    print("=" * 70)
    print("THUNDERSTORM NOWCASTING MODEL TEST")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    batch_size = 2
    time_steps = 4
    channels = 12
    height = 27
    width = 27

    x = torch.randn(
        batch_size,
        time_steps,
        channels,
        height,
        width,
        device=device,
    )

    availability = torch.ones_like(x)

    # Radar + Lightning are unavailable
    x[:, :, 8:, :, :] = float("nan")
    availability[:, :, 8:, :, :] = 0.0

    print("Input shape       :", x.shape)
    print("Availability shape:", availability.shape)
    print("Missing channels  :", "Radar + Lightning")

    model = ThunderstormNowcaster(
        input_channels=channels,
        hidden_channels=64,
        horizons=4,
    ).to(device)

    model.eval()

    with torch.no_grad():

        output = model(
            x,
            availability,
        )

    print("Output shape      :", output.shape)
    print("Output dtype      :", output.dtype)

    print(
        "Output range      :",
        f"{output.min().item():.4f} -> "
        f"{output.max().item():.4f}",
    )

    expected_shape = (
        batch_size,
        4,
        height,
        width,
    )

    assert output.shape == expected_shape

    assert torch.isfinite(output).all()

    print()
    print("MODEL TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_model()