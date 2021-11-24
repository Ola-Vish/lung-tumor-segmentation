import torch
import torchvision
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding='valid'),
            nn.LeakyReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size, padding='valid'),
            nn.LeakyReLU())

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, factor=2, kernel_size=2):
        super().__init__()
        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode='bilinear'),
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding='same'),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.up_block(x)


class Encoder(nn.Module):
    def __init__(self, channels, kernel=2, stride=2):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel, stride)
        self.encoder_blocks = nn.ModuleList(
            [DoubleConvBlock(channels[idx - 1], channels[idx]) for idx in range(1, len(channels))])

    def forward(self, x):
        feature_maps = []
        for block in self.encoder_blocks:
            x = block(x)
            feature_maps.append(x)
            x = self.max_pool(x)
        return feature_maps


class Decoder(nn.Module):
    def __init__(self, channels, kernel_size=2, stride=2):
        super().__init__()
        self.upconvs = nn.ModuleList(
            [UpConvBlock(channels[idx - 1], channels[idx], kernel_size, stride) for idx in
             range(1, len(channels))])
        self.decoder_blocks = nn.ModuleList(
            [DoubleConvBlock(channels[idx - 1], channels[idx]) for idx in range(1, len(channels))])

    def forward(self, x, encoder_features):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            cropped_features = self.crop(encoder_features[i], x)
            x = torch.cat([x, cropped_features], dim=1)
            x = self.decoder_blocks[i](x)
        return x

    def crop(self, existing_map, desired_map):
        batch, channel, hight, width = desired_map.shape
        return torchvision.transforms.CenterCrop((hight, width))(existing_map)


class UNet(nn.Module):
    def __init__(self, enc_chs, dec_chs, num_classes=1,
                 retain_dim=False, out_sz=(572, 572)):
        super().__init__()
        self.encoder = Encoder(channels=enc_chs)
        self.decoder = Decoder(channels=dec_chs)
        self.output = nn.Conv2d(dec_chs[-1], num_classes, 1)
        self.retain_dim = retain_dim
        self.out_size = out_sz

    def forward(self, x):
        encoder_features = self.encoder(x)
        reverse_encoder_features = encoder_features[::-1]
        decoder_output = self.decoder(reverse_encoder_features[0], reverse_encoder_features[1:])
        output = self.output(decoder_output)
        if self.retain_dim:
            output = nn.functional.interpolate(output, self.out_size)
        return output
