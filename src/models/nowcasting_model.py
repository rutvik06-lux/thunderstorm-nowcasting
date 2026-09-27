from pathlib import Path
import torch
import torch.nn as nn


class ReliabilityAwareFusion(nn.Module):
    """
    Fuses 16 predictive meteorological/satellite channels
    with their 16 availability channels.

    LIS is deliberately excluded from the predictive path.
    """

    def __init__(self, input_channels=16, hidden_channels=64):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels * 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, availability):
        # x:              [B, 16, H, W]
        # availability:  [B, 16, H, W]
        fused_input = torch.cat([x, availability], dim=1)
        return self.fusion(fused_input)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, hidden_state):

        h, c = hidden_state

        combined = torch.cat([x, h], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

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
    """
    Multimodal thunderstorm/lightning nowcasting model.

    Predictive input:
        16 channels

    Excluded from prediction:
        LIS lightning observation

    Input:
        x             [B, T, 16, H, W]
        availability [B, T, 16, H, W]

    Output:
        logits        [B, horizons, H, W]
    """

    def __init__(
        self,
        input_channels=16,
        hidden_channels=64,
        horizons=4,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.horizons = horizons

        self.fusion = ReliabilityAwareFusion(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
        )

        self.temporal = ConvLSTMCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                horizons,
                kernel_size=1,
            ),
        )

    def forward(self, x, availability):

        if x.ndim != 5:
            raise ValueError(
                f"Expected x with shape [B,T,C,H,W], got {tuple(x.shape)}"
            )

        if availability.shape != x.shape:
            raise ValueError(
                "Availability tensor must have the same shape as x. "
                f"x={tuple(x.shape)}, availability={tuple(availability.shape)}"
            )

        batch_size, time_steps, channels, height, width = x.shape

        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} predictive channels, "
                f"got {channels}"
            )

        device = x.device

        h, c = self.temporal.init_hidden(
            batch_size,
            height,
            width,
            device,
        )

        for t in range(time_steps):

            fused = self.fusion(
                x[:, t],
                availability[:, t],
            )

            h, c = self.temporal(
                fused,
                (h, c),
            )

        logits = self.decoder(h)

        return logits

    @torch.no_grad()
    def predict_probability(self, x, availability):

        logits = self.forward(x, availability)

        return torch.sigmoid(logits)


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":

    model = ThunderstormNowcaster(
        input_channels=16,
        hidden_channels=64,
        horizons=4,
    )

    print("=" * 70)
    print("THUNDERSTORM NOWCASTER MODEL TEST")
    print("=" * 70)

    print(f"Trainable parameters: {count_parameters(model):,}")

    x = torch.randn(
        1,
        2,
        16,
        27,
        27,
    )

    availability = torch.ones_like(x)

    logits = model(
        x,
        availability,
    )

    probabilities = torch.sigmoid(logits)

    print(f"Input shape:         {tuple(x.shape)}")
    print(f"Availability shape:  {tuple(availability.shape)}")
    print(f"Output logits shape: {tuple(logits.shape)}")
    print(f"Output probability:  {tuple(probabilities.shape)}")

    assert x.shape == (1, 2, 16, 27, 27)
    assert logits.shape == (1, 4, 27, 27)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(probabilities).all()

    assert probabilities.min() >= 0
    assert probabilities.max() <= 1

    print()
    print("LIS predictive channels: EXCLUDED")
    print("Predictive channels:     16")
    print("Forecast horizons:       4")
    print()
    print("MODEL TEST PASSED")
    print("=" * 70)
