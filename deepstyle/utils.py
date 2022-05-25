import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

import pdb


def center_crop_square(im, size):
    width, height = im.size  # Get dimensions

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


#def image_loader(image_name, imsize, device):
# def image_loader(image_name, device, toGray=False):
def image_loader(image_name):
    #loader = transforms.Compose([transforms.CenterCrop( int(0.9*imsize) ),
    #                             transforms.Resize(imsize),
    #                             transforms.ToTensor()])
    # loader = transforms.Compose([transforms.ToTensor()])

    image = Image.open(image_name)
    # #image = center_crop_square(image, min(*image.size))
    # if toGray:
    #     image = Image.fromarray( np.stack([np.array(image.convert("L"))]*3, 2) )

    # # gen batch dimension required to fit network's input dimensions
    # image = loader(image).unsqueeze(0)
    # return image.to(device, torch.float)
    image = image.convert("RGB")
    return image

def image_transform(image, device, toGray=False):
    loader = transforms.Compose([transforms.ToTensor()])
    if toGray:
        image = Image.fromarray( np.stack([np.array(image.convert("L"))]*3, 2) )

    # gen batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)

def get_starting_imgs(args):
    # if args.content is not None:
    #     #content_img = image_loader(args.content, args.imsize, args.device)
    #     content_img = image_loader(args.content, args.device, args.gray)
    #     if args.histmatch:
    #         content_img = hist_match(content_img.cpu().numpy(), style_img.cpu().numpy())
    #         content_img = torch.from_numpy(content_img).to(args.device).float()
    # else:
    #     content_img = None

    # style_img = image_loader(args.style, args.imsize, args.device)
    # style_img = image_loader(args.style, args.device, args.gray)

    # content_img = image_loader(args.content, args.device, args.gray)
    content_img = image_loader(args.content)
    # style_img = image_loader(args.style, args.device, args.gray)
    style_img = image_loader(args.style)
    
    max_wxh = args.gpu_memory*22755
    M, N = content_img.size
    orig_M, orig_N = M, N
    k = 1
    while (M*N > max_wxh):
        M = int(M/k)
        N = int(N/k)
        k += 1

    content_img = content_img.resize((M, N))
    style_img = style_img.resize((M, N))

    content_img = image_transform(content_img, args.device, args.gray)
    style_img = image_transform(style_img, args.device, args.gray)

    if args.histmatch:
        content_img = hist_match(content_img.cpu().numpy(), style_img.cpu().numpy())
        content_img = torch.from_numpy(content_img).to(args.device).float()

    if args.init_img == 'content':
        assert args.content is not None
        gen_img = content_img.clone()
        
    elif args.init_img == 'random':
        gen_img = torch.randn(style_img.data.size(), device=args.device)
        gen_img.data.clamp_(0, 1)
    elif args.init_img == 'style':
        gen_img = style_img.clone()
    else:
        gen_img = Image.open(args.init_img)
        gen_img = gen_img.resize(style_img.size())
        gen_img = F.to_tensor(gen_img).unsqueeze(0).to(args.device)

    #print(gen_img.size())
    return style_img, content_img, gen_img, (orig_M, orig_N)


def save_tensor_img(out_img, outpath, size=None):
    out_img = out_img.cpu().clone()
    # remove the gen batch dimension
    out_img = out_img.squeeze(0)
    out_img = transforms.ToPILImage()(out_img)
    if size:
        out_img = out_img.resize(size)

    out_img.save(outpath)
    return outpath


def plot_losses(losses_dict):
    num_plts = len(losses_dict.keys())
    fig = plt.figure(figsize=(4 * num_plts, 4))
    plot_dims = (1, num_plts)
    for j, k in enumerate(losses_dict.keys()):
        plt.subplot(*plot_dims, 1 + j)
        y = losses_dict[k]
        x = np.arange(len(y))
        plt.plot(x, y)
        plt.title(f"{k} losses")
    return fig


def interpolate(x, y):
    alpha = torch.rand(1) * torch.ones(x.size())
    alpha = alpha.to(x.device)

    return alpha * x.detach() + ((1 - alpha) * y.detach())


import torch.autograd as autograd


def calc_gradient_penalty(f, x):
    x.requires_grad_(True)

    y = f(x)

    grad_outputs = torch.ones(y.size()).to(y.device)
    gradients = autograd.grad(outputs=y, inputs=x, grad_outputs=grad_outputs,
                              create_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sample_k(*xs, k):
    if k is None or k <= 0:
        if len(xs) == 1:
            return xs[0]
        return xs

    n = len(xs[0])
    idcs = np.random.choice(n, min(k, n), replace=False)
    ret = []
    for x in xs:
        assert len(x) == n
        ret.append(x[idcs])

    if len(ret) == 1:
        return ret[0]
    return ret

# histogram matching from source to template/ style image
# Ref: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
