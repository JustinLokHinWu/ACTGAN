import torch
from torch.autograd.variable import Variable

def setup_models(args):
    g = args.generator(args)
    d = args.discriminator(args)
    if args.cuda:
        g = g.cuda()
        d = d.cuda()
    return g,d

def label_to_onehot(labels, num_classes, cuda):
    n_labels = labels.shape[0]
    encoding = torch.zeros((n_labels, num_classes))
    encoding[range(n_labels), labels[range(n_labels)]] = 1

    if cuda:
        encoding = encoding.cuda()
    return encoding

def images_to_vectors(images, args):
    return images.view(images.size(0), args.num_channels * (args.image_size**2))

def vectors_to_images(vectors, args):
    return vectors.view(vectors.size(0), args.num_channels, args.image_size, args.image_size)

def random_labels(batch_size, num_classes, cuda):
    labels = Variable(torch.randint(0, num_classes, (batch_size,)))
    if cuda:
        labels = labels.cuda()
    return labels

def noise(batch_size, noise_size, cuda):
    n = Variable(torch.randn(batch_size, noise_size))
    if cuda: return n.cuda()
    return n

def real_data_target(size, cuda):
    '''
    Tensor containing ones, with shape = size
    '''
    lb = 0.9
    ub = 1.1

    data = Variable(torch.rand(size,1) * (ub - lb) + lb)
    if cuda: return data.cuda()
    return data

def fake_data_target(size, cuda):
    '''
    Tensor containing zeros, with shape = size
    '''
    lb = 0.0
    ub = 0.1

    data = Variable(torch.rand(size,1) * (ub - lb) + lb)
    if cuda: return data.cuda()
    return data

def create_test_samples(args):
    test_noise = noise(args.n_classes, args.noise_size, args.cuda) #passing n classes?

    test_labels = Variable(
        label_to_onehot(torch.arange(10).long(), args.n_classes, args.cuda))
    if args.cuda:
        test_labels = test_labels.cuda()
    return test_noise, test_labels

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def test_model(generator, args):
    logger = Logger(model_name='ACT-GAN-Test', data_name=args.logger_data_name)
    eval_rows = 5

    # Generate inputs for generator
    eval_noise = noise(args.n_classes * eval_rows, args.noise_size, args.cuda)
    eval_labels = torch.arange(args.n_classes).repeat(eval_rows)
    eval_onehot = label_to_onehot(eval_labels, args.n_classes, args.cuda)

    # Generator results
    eval_images = vectors_to_images(generator(eval_noise, eval_onehot), args).data.cpu()

    # # Display Images
    display.clear_output(True)
    logger.log_images(eval_images, 10, 1, 1, 1, n_rows=args.n_classes);