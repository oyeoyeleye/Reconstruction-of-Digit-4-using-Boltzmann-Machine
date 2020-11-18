from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import os
import sys
import time
import matplotlib.pyplot as plt
from mosaic import make_mosaic
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from torchvision.transforms import Compose, ToTensor, Normalize
from collections import defaultdict
from utils import Autoencoder, Discriminator, Padding, lerp, swap_halves, l2_norm, sub2ind


def read_images(num_classes, channels=1, h=32, w=32):
    images = np.zeros((num_classes, channels, h, w), dtype=np.uint8)
    for i in range(num_classes):
        x1 = Image.open(str(i)+'.png')
        x1 = np.expand_dims(np.array(x1)[:, :, 0], axis=0)
        images[i] = x1
    return images


def make_grid(images, num_classes, channels=1, h=32, w=32):
    grid = np.zeros((num_classes*num_classes, channels*2, h, w), dtype=np.uint8)
    for i in range(num_classes):
        for j in range(num_classes):
            grid[sub2ind((num_classes, num_classes), i, j)] = \
                np.concatenate((images[i], images[j]), axis=0)
    return grid



def feed_forward(x, args, device, ae, discriminator, num_devices, opt_ae, opt_d, loss_fn,
                 idx, log_interval):
    x, y = x[0].to(device), x[1].to(device)
    alpha = torch.rand(x.size(0), 1, 1, 1).to(device) / 2
    out, z = ae(x)  # print('x[0]:', x[0].size())
    disc = discriminator(torch.lerp(out, x, args['reg']))

    z_mix = lerp(z, swap_halves(z), alpha)

    out_mix = ae.module.decoder(z_mix) if num_devices > 1 else ae.decoder(z_mix)
    disc_mix = discriminator(out_mix)
    loss_ae = loss_fn(out, x) + l2_norm(disc_mix) * args['advweight']

    opt_ae.zero_grad()
    loss_ae.backward(retain_graph=True)
    opt_ae.step()

    loss_disc = loss_fn(disc_mix, alpha.reshape(-1)) + l2_norm(disc)

    opt_d.zero_grad()
    loss_disc.backward()
    opt_d.step()

    if idx % log_interval == 0:
        return loss_ae.item(), loss_disc.item()
    else:
        return None, None

def main():
    num_devices = torch.cuda.device_count()
    arr = []
    losses = defaultdict(list)
    args = {
        'epochs': 100,
        'width': 32,
        'latent_width': 4,
        'depth': 16,
        'advdepth': 16,
        'advweight': 0.5,
        'reg': 0.2,
        'latent': 2,
        'colors': 1,
        'lr': 0.0001,
        'batch_size': 512*num_devices if num_devices > 1 else 64,
        'device': 'cuda',
        'log_interval': 50
    }
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    ds = MNIST('data', train=True, download=False,
               transform=Compose([Padding(), ToTensor(), Normalize((0.1307,), (0.3081,))]))
    ds_loader = DataLoader(ds, batch_size=args['batch_size'], shuffle=True, **kwargs)

    scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
    ae = Autoencoder(scales, args['depth'], args['latent'], args['colors']).to(args['device'])
    discriminator = Discriminator(scales, args['advdepth'], args['latent'], args['colors']).to(args['device'])

    opt_ae = optim.Adam(ae.parameters(), lr=args['lr'], weight_decay=1e-5)
    opt_d = optim.Adam(discriminator.parameters(), lr=args['lr'], weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    if num_devices > 1:
        ae, discriminator = nn.DataParallel(ae), nn.DataParallel(discriminator)
    ae.to(device), discriminator.to(device)

    try:
        for epoch in tqdm(range(args['epochs'])):
            ae.train(), discriminator.train()
            for idx, x in tqdm(enumerate(ds_loader)):
                loss_ae, loss_disc = feed_forward(
                    x, args, device, ae, discriminator, num_devices, opt_ae, opt_d, loss_fn,
                    idx, args['log_interval'],)
            arr.append(vis2(device, num_devices, ae, arr, class1=2, class2=9, num_alphas=10))
            losses['loss_disc'].append(loss_disc)
            losses['loss_ae'].append(loss_ae)
            print("loss_disc: ", loss_disc)
            print("loss_ae: ", loss_ae)
            ae.train()

    #             if it % 100 == 0:
    #                 img = status()
    #
    #                 plt.figure(facecolor='w', figsize=(10, 4))
    #                 for key in losses:
    #                     total = len(losses[key])
    #                     skip = 1 + (total // 1000)
    #                     y = build_batches(losses[key], skip).mean(axis=-1)
    #                     x = np.linspace(0, total, len(y))
    #                     plt.plot(x, y, label=key, lw=0.5)
    #                 plt.legend(loc='upper right')
    #
    #                 plt.show()
    #                 show_array(img * 255)
    #
    #                 speed = args['batch_size'] * it / (time.time() - start_time)
    #                 print(f'{epoch + 1}/{args["epochs"]}; {speed:.2f} samples/sec')
    #
    #             it += 1
    except KeyboardInterrupt:
        pass
    with open('losses.pickle', 'wb') as handle:
        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('arr.pickle', 'wb') as handle:
        pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # # In[8]:
    #
    # # show the distribution of predictions from the discriminator
    # plt.hist(disc_mix.data.cpu().numpy(), range=[0, 0.5], bins=20)
    # plt.show()
    # print(disc_mix)
    #
    # # In[11]:
    #
    # # distribution of each z dimension
    # z = encoder(x_batches[0])
    # z = z.data.cpu().numpy().reshape(len(z), -1).T
    # for dim in z:
    #     plt.hist(dim, bins=12, alpha=0.1)
    # plt.show()


if __name__ == "__main__":
    # main()
    # with open("arr.pickle", "rb") as f:
    #     epochs = pickle.load(f)
    #     epochs = [item for item in epochs if type(item) != list]
    #     print(len(epochs))
    #
    #     for k in range(200):
    #         fig, ax = plt.subplots(nrows=5, ncols=4, sharex=True, sharey=True)
    #         for i in tqdm(range(5)):
    #             for j in range(4):
    #                 ax[i, j].imshow(epochs[k][sub2ind((5, 4), i, j)].squeeze())
    #                 ax[i, j].get_xaxis().set_ticks([])
    #                 ax[i, j].get_yaxis().set_ticks([])
    #                 # ax[i, j].title.set_text('2-9')
    #         fig.suptitle('epoch={}'.format(k),  y=0.98, fontsize=9)
    #         plt.savefig('Images/epochs{}.png'.format(k))
    #         # plt.show()

    images = read_images(num_classes=10, channels=1, h=32, w=32)
    images = make_grid(images, num_classes=10, channels=1, h=32, w=32)
    print(images.shape)