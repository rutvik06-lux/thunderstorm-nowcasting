import torch
import torch.nn as nn


class ReliabilityAwareFusion(nn.Module):
    """
    Fuses multimodal atmospheric observations with
    per-channel availability information.

    Input:
        x            [B, C, H, W]
        availability [B, C, H, W]

    Output:
        [B, 64, H, W]
    """

    def __init__(self, feature_channels=18):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                feature_channels * 2,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, availability):

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        availability = torch.nan_to_num(
            availability,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        fused_input = torch.cat(
            [x, availability],
            dim=1,
        )

        return self.encoder(fused_input)


class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
    ):
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

        combined = torch.cat(
            [x, h_prev],
            dim=1,
        )

        gates = self.gates(combined)

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=1,
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c

    def init_hidden(
        self,
        batch_size,
        height,
        width,
        device,
    ):

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

    Input:
        x:
            [B, T, 18, H, W]

        availability:
            [B, T, 18, H, W]

    Output:
        logits:
            [B, 4, H, W]

    Four forecast horizons:
        +30 min
        +60 min
        +90 min
        +120 min

    Apply sigmoid to logits during inference to obtain
    lightning probabilities.
    """

    def __init__(
        self,
        input_channels=18,
        hidden_channels=64,
        horizons=4,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.horizons = horizons

        self.fusion = ReliabilityAwareFusion(
            feature_channels=input_channels
        )

        self.convlstm = ConvLSTMCell(
            input_channels=64,
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
                "Expected x with shape "
                "[B, T, C, H, W]"
            )

        if availability.shape != x.shape:
            raise ValueError(
                "Availability must have the same shape "
                "as x."
            )

        batch_size, time_steps, channels, height, width = (
            x.shape
        )

        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input "
                f"channels, received {channels}."
            )

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

        logits = self.decoder(h)

        return logits


def test_model():

    print("=" * 72)
    print("MULTIMODAL THUNDERSTORM NOWCASTING MODEL TEST")
    print("=" * 72)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    batch_size = 1
    time_steps = 2
    channels = 18
    height = 27
    width = 27
    horizons = 4

    x = torch.randn(
        batch_size,
        time_steps,
        channels,
        height,
        width,
        device=device,
    )

    availability = torch.ones_like(x)

    # Simulate unavailable IMERG and LIS observations.
    x[:, :, 5, :, :] = float("nan")
    x[:, :, 16, :, :] = float("nan")

    availability[:, :, 5, :, :] = 0.0
    availability[:, :, 16, :, :] = 0.0

    print("Input shape       :", x.shape)
    print("Availability shape:", availability.shape)
    print("Missing channels  :", "IMERG + LIS")

    model = ThunderstormNowcaster(
        input_channels=channels,
        hidden_channels=64,
        horizons=horizons,
    ).to(device)

    model.eval()

    with torch.no_grad():

        logits = model(
            x,
            availability,
        )

        probabilities = torch.sigmoid(
            logits
        )

    print("Logits shape      :", logits.shape)
    print("Probability shape :", probabilities.shape)
    print(
        "Probability range :",
        f"{probabilities.min().item():.4f}"
        f" -> "
        f"{probabilities.max().item():.4f}",
    )

    expected_shape = (
        batch_size,
        horizons,
        height,
        width,
    )

    assert logits.shape == expected_shape
    assert probabilities.shape == expected_shape

    assert torch.isfinite(logits).all()
    assert torch.isfinite(probabilities).all()

    assert (
        probabilities.min().item() >= 0.0
    )

    assert (
        probabilities.max().item() <= 1.0
    )

    print()
    print("Output shape       : PASS")
    print("Logits finite      : PASS")
    print("Probabilities valid: PASS")
    print()
    print("MODEL TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    test_model()
