# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch
from torch.autograd import Variable, grad
from torch.nn.init import xavier_normal
from torchvision import datasets, transforms
import torchvision.utils as vutils

def xavier_init(model):
    for param in model.parameters():
        if len(param.size()) == 2:
            xavier_normal(param)


if __name__ == '__main__':
    batch_size = 128
    z_dim = 100
    h_dim = 128
    y_dim = 784
    max_epochs = 1000
    lambda_ = 10

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=False, drop_last=True)

    generator = torch.nn.Sequential(torch.nn.Linear(z_dim, h_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h_dim, y_dim),
            torch.nn.Sigmoid())

    discriminator = torch.nn.Sequential(torch.nn.Linear(y_dim, h_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h_dim, 1),
            torch.nn.Sigmoid())

    # Init weight matrices (xavier_normal)
    xavier_init(generator)
    xavier_init(discriminator)

    opt_g = torch.optim.Adam(generator.parameters())
    opt_d = torch.optim.Adam(discriminator.parameters())

    criterion = torch.nn.BCELoss()
    X = Variable(torch.FloatTensor(batch_size, y_dim))
    z = Variable(torch.FloatTensor(batch_size, z_dim))
    labels = Variable(torch.FloatTensor(batch_size))

    # Train
    for epoch in range(max_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            X.data.copy_(data)

            # Update discriminator
            # train with real
            discriminator.zero_grad()
            pred_real = discriminator(X)
            labels.data.fill_(1.0)
            loss_d_real = criterion(pred_real, labels)
            loss_d_real.backward()

            # train with fake
            z.data.normal_(0, 1)
            fake = generator.forward(z).detach()
            pred_fake = discriminator(fake)
            labels.data.fill_(0.0)
            loss_d_fake = criterion(pred_fake, labels)
            loss_d_fake.backward()

            # gradient penalty
            alpha = torch.rand(batch_size, 1).expand(X.size())
            x_hat = Variable(alpha * X.data + (1 - alpha) * (X.data + 0.5 * X.data.std() * torch.rand(X.size())), requires_grad=True)
            pred_hat = discriminator(x_hat)
            gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            gradient_penalty.backward()

            loss_d = loss_d_real + loss_d_fake + gradient_penalty
            opt_d.step()

            # Update generator
            generator.zero_grad()
            z.data.normal_(0, 1)
            gen = generator(z)
            pred_gen = discriminator(gen)
            labels.data.fill_(1.0)
            loss_g = criterion(pred_gen, labels)
            loss_g.backward()
            opt_g.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, max_epochs, batch_idx, len(train_loader),
                     loss_d.data[0], loss_g.data[0]))

            if batch_idx % 100 == 0:
                vutils.save_image(data,
                        'samples/real_samples.png')
                fake = generator(z)
                vutils.save_image(gen.data.view(batch_size, 1, 28, 28),
                        'samples/fake_samples_epoch_%03d.png' % epoch)


