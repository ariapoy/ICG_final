import argparse

parser = argparse.ArgumentParser(description='Style Transfer')

# Style Loss
parser.add_argument('--distance', type=str, default='wgan-gp',
                    choices=['wgan-gp', 'sngan', 'wgan-sn', 'quad', 'linear', 'gauss', 'norm', 'gram'])
parser.add_argument('--samples', type=int, default=1024,
                    help='number of features to sample from for each layer per training step. if set to 0, all features are used')

# Training
parser.add_argument('--steps', type=int, default=1000, help='num training steps')
parser.add_argument('--img-lr', type=float, default=1e-2,
                    help='learning rate for image pixels')
parser.add_argument('--disc-lr', type=float, default=1e-2,
                    help='learning rate for discriminators')
parser.add_argument('--disc-l2', type=float, default=0,
                    help='weight decay')
parser.add_argument('--opt', choices=['adam', 'sgd'], default='adam', help='optimizer')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='style-content balance ratio. larger values weigh style more. ' \
                         'if doing style representation (no content), then this value is ignored')
parser.add_argument('--weight_S0', type=float, default=0.166666667, help='weight of zeroth style loss')
parser.add_argument('--weight_S1', type=float, default=0.166666667, help='weight of first style loss')
parser.add_argument('--weight_S2', type=float, default=0.166666667, help='weight of second style loss')
parser.add_argument('--weight_S3', type=float, default=0.166666667, help='weight of third style loss')
parser.add_argument('--weight_S4', type=float, default=0.166666667, help='weight of forth style loss')
parser.add_argument('--weight_S5', type=float, default=0.166666667, help='weight of fifth style loss')

parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')

# CNN
parser.add_argument('--cnn', type=str, default='vgg19-bn',
                    choices=['vgg19-bn', 'vgg19-bn-relu', 'vgg19', 'vgg19-relu', 'resnet18', 'dense121'])
parser.add_argument('--layers', type=int, default=5, help='number of layers to. should be within [0, 5]')
parser.add_argument('--disc-hdim', type=int, default=256, help='dimension of the hidden layers in the discriminator')
parser.add_argument('--random', dest='pretrained', action='store_false')

# Images
parser.add_argument('--gpu_memory', type=int, default=8,
                    help='gpu memory size we can use.')
parser.add_argument('--init-img', type=str, default='content',
                    help='how to initialize the generated image. can be one of [random, content, <path to image>]')
parser.add_argument('--style', type=str, help='path to style image')
parser.add_argument('--content', type=str, default=None, help='optional path to content image')
parser.add_argument('--gray', type=str, default="False", help='RGB to Gray')
parser.add_argument('--histmatch', type=str, default="False", help='Histogram match')

# Output
parser.add_argument('--out-dir', type=str, default='out/', help='directory to save all work')
parser.add_argument('--out_name', type=str, default='gen', help='name of output')
#parser.add_argument('--gif-frame', type=int, default=100,
#                    help='interval to save the generated image w.r.t to the training steps to make a GIF of the style transfer')

import os
import utils
import arch
from arch import cnn
import style

import pdb

def run(args):
    # Images
    style_img, content_img, gen_img, orig_content_size = utils.get_starting_imgs(args)

    # CNN layers
    style_layers, content_layers = cnn.get_layers(args)

    # Make model
    model = arch.make_model(args, style_layers, content_layers, style_img, content_img)

    # Transfer
    losses_dict, gen_hist = style.transfer(args, gen_img, style_img, model)

    # Plot losses
    loss_fig = utils.plot_losses(losses_dict)

    # Save
    content_name = args.content.split('/')[-1]
    content_name = content_name.split('.')[0]
    style_name = args.style.split('/')[-1]
    style_name = style_name.split('.')[0]
    # gen_name = args.out_name
    gen_name = args.out_name.replace("img-style/", "")
    # create out-dir
    from pathlib import Path
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    # content img
    utils.save_tensor_img(content_img, os.path.join(args.out_dir, '{0}-{1}.png'.format(content_name, style_name)))
    # style img
    utils.save_tensor_img(style_img, os.path.join(args.out_dir, '{}.png'.format(style_name)))
    # Generated image
    utils.save_tensor_img(gen_img, os.path.join(args.out_dir, 'gen_{}.png'.format(gen_name)))
    # Generated image resize
    utils.save_tensor_img(gen_img, os.path.join(args.out_dir, '{}-s1.png'.format(gen_name)), orig_content_size)
    #gen_hist[0].save(os.path.join(args.out_dir, 'gen.gif'), save_all=True, append_images=gen_hist[1:])
    # Losses
    #loss_fig.savefig(os.path.join(args.out_dir, 'losses.png'))
    print(f"Results saved to '{args.out_dir}'")


if __name__ == '__main__':
    args = parser.parse_args()
    args.gray, args.histmatch = eval(args.gray), eval(args.histmatch)
    print(args)
    run(args)
