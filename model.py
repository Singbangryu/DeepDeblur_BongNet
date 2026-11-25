import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)

    def forward(self, x):
        x_in = x
        x = self.conv1(x_in)
        x = self.relu(x)
        x = self.conv2(x)
        return x_in + x


class BongNet(nn.Module):
    def __init__(self):
        super().__init__()

        # PixelUnshuffle packs RGB neighborhoods into 12 and 48 channels.
        self.fine_root = nn.Conv2d(12, 64, 3, padding=1)
        self.coarse_root = nn.Conv2d(48, 128, 3, padding=1)

        self.fine_path1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
        )

        self.coarse_path1 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
        )

        self.fine_to_coarse_1 = nn.Conv2d(64, 128, 2, stride=2)
        self.coarse_to_fine_1 = nn.Conv2d(128, 64, 1)

        self.fine_path2 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )
        self.coarse_path2 = nn.Sequential(
            ResBlock(128),
            ResBlock(128)
        )

        self.fine_to_coarse_2 = nn.Conv2d(64, 128, 2, stride=2)
        self.coarse_to_fine_2 = nn.Conv2d(128, 64, 1)

        self.fine_path3 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )
        self.coarse_path3 = nn.Sequential(
            ResBlock(128),
            ResBlock(128)
        )

        self.fine_to_coarse_3 = nn.Conv2d(64, 128, 2, stride=2)
        self.coarse_to_fine_3 = nn.Conv2d(128, 64, 1)

        self.fine_path4 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )
        self.coarse_path4 = nn.Sequential(
            ResBlock(128),
            ResBlock(128)
        )

        self.coarse_to_fine_4 = nn.Conv2d(128, 64, 1)

        self.output_path = nn.Conv2d(64, 12, 3, padding=1)
        self.gate_path = nn.Conv2d(64, 4, 3, padding=1)

        nn.init.zeros_(self.output_path.weight)
        nn.init.zeros_(self.output_path.bias)
        nn.init.zeros_(self.gate_path.weight)
        nn.init.zeros_(self.gate_path.bias)

    def forward(self, blur, debug=False):
        h, w = blur.shape[-2:]
        pad_size = (0, (-w) % 4, 0, (-h) % 4)
        x = F.pad(input=blur, pad=pad_size, mode="replicate")
        fine = F.pixel_unshuffle(x, 2)
        coarse = F.pixel_unshuffle(x, 4)
        fine = self.fine_root(fine)
        coarse = self.coarse_root(coarse)
        fine = F.relu(fine)
        coarse = F.relu(coarse)

        fine = self.fine_path1(fine)
        coarse = self.coarse_path1(coarse)

        up = F.interpolate(
            self.coarse_to_fine_1(coarse),
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        down = self.fine_to_coarse_1(fine)

        fine = fine + up
        coarse = coarse + down
        fine = self.fine_path2(fine)
        coarse = self.coarse_path2(coarse)

        up = F.interpolate(
            self.coarse_to_fine_2(coarse),
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        down = self.fine_to_coarse_2(fine)

        fine = fine + up
        coarse = coarse + down

        fine = self.fine_path3(fine)
        coarse = self.coarse_path3(coarse)

        up = F.interpolate(
            self.coarse_to_fine_3(coarse),
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        down = self.fine_to_coarse_3(fine)

        fine = fine + up
        coarse = coarse + down

        fine = self.fine_path4(fine)
        coarse = self.coarse_path4(coarse)

        up = F.interpolate(
            self.coarse_to_fine_4(coarse),
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        fine = fine + up

        output = F.pixel_shuffle(self.output_path(fine), 2)
        deblur = output[..., :h, :w]

        gate = F.pixel_shuffle(self.gate_path(fine), 2)
        gate = gate[..., :h, :w]

        gate = torch.sigmoid(gate)
        deblurred = blur - gate * deblur

        if debug:
            return deblurred, gate, deblur
        return deblurred
