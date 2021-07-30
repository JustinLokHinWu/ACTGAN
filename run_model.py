import argparse

from torchvision import transforms
from models.generator import ACGAN_Generator
from attrdict import AttrDict
import json
import torch
import glob
from PIL import Image
from models.generator import ACGAN_Generator
from utils.helpers import noise, label_to_onehot

class GeneratorRunner:
    '''
    cfg: AttrDict containing model parameters
    '''
    def __init__(self, cfg):
        self.generator = ACGAN_Generator(cfg)
        self.generator.eval()
        if cfg.cuda:
            self.generator = self.generator.cuda()

        self.models_dir = cfg.models_dir
        self.noise_size = cfg.noise_size
        self.cuda = cfg.cuda
        self.n_classes = cfg.n_classes
        self.valid_epochs = None

    '''
    Generates an image of class class_id using a generator at epoch epoch,
    using a custom seed if supplied. Returns None on failure.
    class_id: int indicating image class
    seed: int for pytorch seed
    '''
    def evaluate(self, class_id, epoch, seed=None):
        if class_id < 0 or class_id >= self.n_classes:
            return None

        try:
            t = torch.load('{}/G_epoch_{}'.format(self.models_dir, epoch))
            self.generator.load_state_dict(t)
        except:
            return None
        
        if seed:
            torch.manual_seed(seed)

        # Setup input for generator
        input_noise = noise(1, self.noise_size, self.cuda)
    
        input_onehot = label_to_onehot(
            torch.Tensor([class_id]).long(),
            self.n_classes, self.cuda
        )

        with torch.no_grad():
            output = self.generator(input_noise, input_onehot) / 2.0 + 0.5

        image = transforms.ToPILImage()(output[0].cpu())
        return image

    '''
    Return a list of all valid generator epochs in models_dir
    '''
    def get_valid_epochs(self):
        if self.valid_epochs is None:
            paths = glob.glob('{}/G_epoch_*'.format(self.models_dir))
        
            self.valid_epochs = sorted(
                [int(path.rsplit('_', 1)[1]) for path in paths])
        return self.valid_epochs

if __name__=='__main__':
    # Load arguments
    parser = argparse.ArgumentParser(
        description='Load and run pretrained ACTGAN generator.'
    )
    parser.add_argument(
        '--config',
        help='path to config json used to train the generator',
        required=True
    )
    parser.add_argument(
        '--out_dir',
        help='path to output file',
        required=True
    )
    parser.add_argument(
        '--class_id',
        help='class index for generated image',
        required=True,
        type=int
    )
    parser.add_argument(
        '--epoch',
        help='model epoch to use',
        type=int
    )  
    parser.add_argument(
        '--seed',
        help='seed for random noise generation',
        type=int
    )  

    args = parser.parse_args()
    
    # Load config from file
    with open(args.config) as f:
        cfg = AttrDict(json.load(f))

    gen_runner = GeneratorRunner(cfg)

    if args.epoch is not None:
        epoch = args.epoch
    else:
        epoch = gen_runner.get_valid_epochs()[-1]

    image = gen_runner.evaluate(args.class_id, epoch=epoch, seed=args.seed)

    image.save(args.out_dir)
    