import torch
from torch import nn, optim
import numpy as np

from utils.logger import Logger
from utils.helpers import *
import utils.helpers

def train(cfg, dataloader):
    logger = Logger(cfg.models_dir)

    generator, discriminator = utils.helpers.setup_models(cfg)
    generator.apply(weights_init)

    # Optimizers
    opt_g = optim.Adam(generator.parameters(), cfg.g_lr)
    opt_d = optim.Adam(discriminator.parameters(), cfg.d_lr)

    # Loss functions
    loss_C = nn.NLLLoss()
    loss_S = nn.BCELoss()

    # Constant test noise
    test_noise, test_labels = create_test_samples(cfg)

    num_batches = len(dataloader)

    for epoch in range(cfg.num_epochs):
        for batch, (real_batch, real_labels) in enumerate(dataloader):
            if cfg.cuda:
                real_batch = real_batch.cuda()
                real_labels = real_labels.cuda()
            
            # Train discriminator on real data
            discriminator.zero_grad()
            disc_real_s, disc_real_c = discriminator(real_batch)

            ones_S = real_data_target(cfg.batch_size, cfg.cuda)

            disc_real_error_s = loss_S(disc_real_s, ones_S)
            disc_real_error_c = loss_C(disc_real_c, real_labels)
            disc_real_error = disc_real_error_s  + disc_real_error_c

            disc_real_error.backward()

            # Train discriminator on generated data
            disc_gen_noise = noise(cfg.batch_size, cfg.noise_size, cfg.cuda)
            disc_gen_rand_labels = random_labels(cfg.batch_size, cfg.n_classes, cfg.cuda)
            disc_gen_rand_onehot = label_to_onehot(disc_gen_rand_labels, cfg.n_classes, cfg.cuda)
            disc_gen_output = generator(disc_gen_noise, disc_gen_rand_onehot)
            disc_fake_s, disc_fake_c = discriminator(disc_gen_output.detach())
            
            zeros_S = fake_data_target(cfg.batch_size, cfg.cuda)

            disc_fake_error_s = loss_S(disc_fake_s, zeros_S)
            disc_fake_error_c = loss_C(disc_fake_c, disc_gen_rand_labels)
            disc_fake_error = disc_fake_error_s + disc_fake_error_c
            disc_fake_error.backward()

            disc_error = disc_real_error + disc_fake_error
            opt_d.step()

            # Train generator
            generator.zero_grad()
            gen_s, gen_c = discriminator(disc_gen_output)
            gen_error_s = loss_S(gen_s, ones_S)
            gen_error_c = loss_C(gen_c, disc_gen_rand_labels)
            gen_error = gen_error_s + gen_error_c 

            gen_error.backward()
            opt_g.step()

            # For last batch in epoch, log loss metrics
            if batch == num_batches - 1:
                logger.log_losses('Discriminator', {
                    'Discriminator Loss': disc_error,
                    'Discriminator Real Class Loss': disc_real_error_c,
                    'Discriminator Real Source Loss': disc_real_error_s,
                    'Discriminator Fake Class Loss': disc_fake_error_c,
                    'Discriminator Fake Source Loss': disc_fake_error_s,
                }, epoch)
                logger.log_losses('Generator', {
                    'Generator Loss': gen_error,
                    'Generator Class Loss': gen_error_c,
                    'Generator Source Loss': gen_error_s
                }, epoch)

        test_images = generator(test_noise, test_labels)
        logger.log_images('Test Images', test_images, epoch)
        if epoch % cfg.epoch_save_rate == 0:
            logger.save_models(generator, discriminator, epoch)

    logger.close()
    return generator, discriminator
