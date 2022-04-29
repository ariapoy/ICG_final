import torch
from torch import nn

from arch import layers, kernels


class TransferModel(nn.Module):
    def __init__(self, args, style_layers, style_img):
        super().__init__()

        self.distance = args.distance
        if 'gan' in args.distance:
            self.layer_type = 'disc'
        else:
            self.layer_type = 'kernel'

        # Style
        main = []
        style_feat = style_img
        for cnn_layer in style_layers:
            with torch.no_grad():
                style_feat = cnn_layer(style_feat)
            assert style_feat.requires_grad == False

            if self.layer_type == 'disc':
                main.append(layers.StyleLayerDisc(args.distance, cnn_layer, style_feat.shape[1], args.samples,
                                                  args.disc_hdim))
            else:
                assert self.layer_type == 'kernel'
                assert args.samples is not None
                kernel = kernels.kernel_map[args.distance]
                main.append(layers.StyleLayerKernel(cnn_layer, style_feat, kernel,
                                                    args.samples))
        self.style = nn.Sequential(*main)
        self.style_weights = [args.weight_S0, 
                              args.weight_S1, 
                              args.weight_S2, 
                              args.weight_S3, 
                              args.weight_S4, 
                              args.weight_S5 
                             ]

    def configure_content(self, content_layers, content_img):
        # Content
        self.content = nn.Sequential(*content_layers)
        with torch.no_grad():
            self.content_feat = self.content(content_img)
            self.content_feat.requires_grad_(False)

    def forward(self, img):
        if hasattr(self, 'content'):
            semantic_feat = self.content(img)
            content_loss = torch.mean((semantic_feat - self.content_feat) ** 2)
        else:
            content_loss = torch.tensor(0.0)

        _, style_losses = self.style((img, []))
        ret = 0
        for idx in range(len(style_losses)):
            ret += style_losses[idx] * self.style_weights[idx]
        return ret, content_loss

    def disc_gp(self, x):
        gp_sum = 0
        for disc in self.style.children():
            x, gp = disc.disc_gp(x)
            gp_sum += gp
        return gp_sum / len(self.style)

    def conv_parameters(self):
        params = []
        for disc_layer in self.style.children():
            params.extend(list(disc_layer.conv.parameters()))
        return params

    def disc_parameters(self):
        params = []
        for disc_layer in self.style.children():
            params.extend(list(disc_layer.disc.parameters()))
        return params


def make_model(args, style_layers, content_layers, style_img, content_img):
    # Initialize model
    model = TransferModel(args, style_layers, style_img).to(args.device)
    #print(model)

    # Freeze CNN
    for params in model.conv_parameters():
        params.requires_grad = False

    # Configure Content
    if args.content is not None:
        model.configure_content(content_layers, content_img)

    return model
