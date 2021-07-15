import torch
from torch import nn, optim

from utils.logger import Logger
from utils.helpers import *
import utils.helpers

def train(cfg, dataloader):
    logger = Logger(model_name='ACGAN-Transformer', data_name=cfg.logger_data_name)

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
                real_labels_f = real_labels.float()
            
            # Convert labels to onehot encoding
            real_onehot = label_to_onehot(real_labels, cfg.n_classes, cfg.cuda)

            # Train discriminator on real data
            opt_d.zero_grad()
            disc_real_s, disc_real_c = discriminator(real_batch)

            ones_S = real_data_target(cfg.batch_size, cfg.cuda)
            disc_real_error = loss_S(disc_real_s, ones_S) + loss_C(disc_real_c, real_labels)

            disc_real_error.backward()

            # Train discriminator on generated data
            disc_gen_noise = noise(cfg.batch_size, cfg.noise_size, cfg.cuda)
            disc_gen_rand_labels = random_labels(cfg.batch_size, cfg.n_classes, cfg.cuda)
            disc_gen_rand_onehot = label_to_onehot(disc_gen_rand_labels, cfg.n_classes, cfg.cuda)
            disc_gen_output = generator(disc_gen_noise, disc_gen_rand_onehot)
            disc_fake_s, disc_fake_c = discriminator(disc_gen_output.detach())
            
            zeros_S = fake_data_target(cfg.batch_size, cfg.cuda)
            disc_fake_error = loss_S(disc_fake_s, zeros_S) + loss_C(disc_fake_c, disc_gen_rand_labels)
            disc_fake_error.backward()

            disc_error = disc_real_error + disc_fake_error
            opt_d.step()

            # Train generator
            generator.zero_grad()
            gen_s, gen_c = discriminator(disc_gen_output)
            gen_error = loss_S(gen_s, ones_S) + loss_C(gen_c, disc_gen_rand_labels)

            gen_error.backward()
            opt_g.step()

            # if (batch) % 100 == 0:
            #     display.clear_output(True)
            #     # Display Images
            #     test_images = vectors_to_images(generator(test_noise, test_labels), cfg).data.cpu()
            #     logger.log_images(test_images, cfg.n_classes, epoch, batch, num_batches, cfg.n_classes);
            #     # Display status Logs
            #     logger.display_status(
            #         epoch, cfg.num_epochs, batch, num_batches,
            #     disc_error, gen_error, disc_real_error, disc_fake_error
            #     )
    #     # Model Checkpoints
    #     logger.save_models(generator, discriminator, epoch)
    # return generator, discriminator
    1==1