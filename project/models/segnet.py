import torch
import torchvision
import torch.nn as nn


class VggSubBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(output_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU())

    def forward(self, x):
        return self.block(x)


class VggBlock(nn.Module):
    def __init__(self, input_channels, output_channels, repetitions=2):
        super().__init__()
        self.first_block = VggSubBlock(input_channels, output_channels)
        self.remaining_blocks = nn.Sequential(
            *[VggSubBlock(output_channels, output_channels) for _ in range(1, repetitions)])

    def forward(self, x):
        x = self.first_block(x)
        x = self.remaining_blocks(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, repetitions=2):
        super().__init__()
        self.first_blocks = nn.Sequential(
            *[VggSubBlock(input_channels, input_channels) for _ in range(1, repetitions)])
        self.last_block = VggSubBlock(input_channels, output_channels)

    def forward(self, x):
        x = self.first_blocks(x)
        x = self.last_block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True,
                                     ceil_mode=False)
        self.encoder_blocks = nn.ModuleList(
            [VggBlock(channels[idx - 1], channels[idx], repetitions=2) if idx < 3 else VggBlock(channels[idx - 1],
                                                                                                channels[idx],
                                                                                                repetitions=3) for idx
             in range(1, len(channels))])

    def forward(self, x):
        pool_indices = []
        for block in self.encoder_blocks:
            x = block(x)
            x, indices = self.max_pool(x)
            pool_indices.append(indices)
        return x, pool_indices


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(channels[idx - 1], channels[idx], repetitions=2) if idx > 3 else DecoderBlock(
                channels[idx - 1], channels[idx], repetitions=3) for idx in range(1, len(channels))])

    def forward(self, x, pool_indices):
        for idx, block in enumerate(self.decoder_blocks):
            x = self.max_unpool(x, pool_indices[idx])
            x = block(x)
        return x


class SegNet(nn.Module):
    def __init__(self, enc_chs, dec_chs, num_classes=2, warm_start=True):
        super().__init__()
        self.encoder = Encoder(channels=enc_chs)
        self.decoder = Decoder(channels=dec_chs)
        self.last = VggSubBlock(dec_chs[-1], dec_chs[-1])
        self.output = nn.Sequential(
            nn.Conv2d(dec_chs[-1], num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(num_classes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        if warm_start:
            self.load_vgg_weights_to_encoder()

    def forward(self, x):
        encoder_features, pool_indices = self.encoder(x)
        reverse_pool_indices = pool_indices[::-1]
        decoder_output = self.decoder(encoder_features, reverse_pool_indices)
        decoder_output = self.last(decoder_output)
        decoder_output = self.output(decoder_output)
        return decoder_output

    def load_vgg_weights_to_encoder(self):
        encoder_state_dict = self.encoder.state_dict()
        encoder_keys = list(encoder_state_dict.keys())
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        vgg_state_dict = vgg16.state_dict()
        for idx, key in enumerate(vgg_state_dict):
            if idx < len(encoder_keys):
                curr_key = encoder_keys[idx]
                curr_value = vgg_state_dict[key]
                encoder_state_dict[curr_key] = curr_value
        self.encoder.load_state_dict(encoder_state_dict)