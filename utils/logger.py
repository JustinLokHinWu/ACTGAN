import torch
import torchvision
import os
import errno

from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, models_dir):
        self.writer = SummaryWriter()
        self.name = self.writer.get_logdir().split("\\")[1]
        self.models_dir = models_dir

    def log_images(self, tag, images, epoch):
        transformed_images = (images / 2.0) + 0.5
        self.writer.add_images(tag, transformed_images, epoch)

    # def log_losses(self, gen_loss, disc_loss, epoch):
    #     self.writer.add_scalar('Generator loss', gen_loss, epoch)
    #     self.writer.add_scalar('Discriminator loss', disc_loss, epoch)

    def log_losses(self, tag, losses, epoch):
        self.writer.add_scalars(tag, losses, epoch)

    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.name)
        try:
            os.makedirs(out_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()