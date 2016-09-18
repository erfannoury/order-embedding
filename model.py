"""
Model specification
"""
import theano
import theano.tensor as tensor
from theano.tensor.extra_ops import fill_diagonal

from collections import OrderedDict

from utils import _p, ortho_weight, norm_weight, xavier_weight, tanh, l2norm
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer


def init_params(options):
    """
    Initialize all parameters
    """
    params = OrderedDict()

    # Word embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

    # Sentence encoder
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])

    # Image encoder
    params = get_layer('ff')[0](options, params, prefix='ff_image', nin=options[
        'dim_image'], nout=options['dim'])

    # Discriminator
    params = get_layer('ff')[0](
        options, params, prefix='ff_disc', nin=options['dim'], nout=1)

    return params


def order_violations(s, im, options):
    """
    Computes the order violations (Equation 2 in the paper)
    """
    return tensor.pow(tensor.maximum(0, s - im), 2)


def discriminate(tparams, emb, options):
    """
    Apply discriminator to the iamge and sentence embeddings
    """
    return get_layer('ff')[1](tparams, emb, options,
                              prefix='ff_disc', activ='sigmoid')


def discriminator_loss(s, im, options):
    """
    Computes the disciminator loss
    """
    s_disc = tensor.clip(s, 1e-6, 1. - 1e-6)
    im_disc = tensor.clip(im, 1e-6, 1. - 1e-6)
    im_loss = tensor.nnet.binary_crossentropy(
        im_disc, tensor.zeros_like(im_disc)).mean()
    s_loss = tensor.nnet.binary_crossentropy(
        s_disc, tensor.ones_like(s_disc)).mean()
    return -options['disc_coeff'] * (im_loss + s_loss)


def contrastive_loss(s, im, options):
    """
    For a minibatch of sentence and image embeddings, compute the pairwise contrastive loss
    """
    margin = options['margin']

    if options['method'] in ['order', 'order_discriminator']:
        im2 = im.dimshuffle(('x', 0, 1))
        s2 = s.dimshuffle((0, 'x', 1))
        errors = order_violations(s2, im2, options).sum(axis=2)
    elif options['method'] == 'cosine':
        # negative because error is the opposite of (cosine) similarity
        errors = - tensor.dot(im, s.T)

    diagonal = errors.diagonal()

    # compare every diagonal score to scores in its column (all contrastive
    # images for each sentence)
    cost_s = tensor.maximum(0, margin - errors + diagonal)
    # all contrastive sentences for each image
    cost_im = tensor.maximum(0, margin - errors + diagonal.reshape((-1, 1)))

    cost_tot = cost_s + cost_im

    # clear diagonals
    cost_tot = fill_diagonal(cost_tot, 0)

    return cost_tot.sum()


def encode_sentences(tparams, options, x, mask):
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape(
        [n_timesteps, n_samples, options['dim_word']])

    # Encode sentences (source)
    proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                            prefix='encoder',
                                            mask=mask)
    s = l2norm(proj[0][-1])
    if options['abs']:
        s = abs(s)

    return s


def encode_images(tparams, options, im):
    im_emb = get_layer('ff')[1](tparams, im, options,
                                prefix='ff_image', activ='linear')
    im_emb = l2norm(im_emb)
    if options['abs']:
        im_emb = abs(im_emb)

    return im_emb


def build_model(tparams, options):
    """
    Computation graph for the entire model
    """
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    im = tensor.matrix('im', dtype='float32')

    # embed sentences and images
    s_emb = encode_sentences(tparams, options, x, mask)
    im_emb = encode_images(tparams, options, im)

    # discriminate sentence and image embeddings
    s_disc = discriminate(tparams, s_emb, options)
    im_disc = discriminate(tparams, im_emb, options)

    # Contrastive loss
    ccost = contrastive_loss(s_emb, im_emb, options)

    # Discriminatore loss
    dcost = discriminator_loss(s_disc, im_disc, options)

    # Total cost
    cost = ccost + dcost

    return [x, mask, im], cost, [ccost, dcost]


def build_sentence_encoder(tparams, options):
    """
    Encoder only, for sentences
    """
    # sentence features
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')

    return [x, mask], encode_sentences(tparams, options, x, mask)


def build_image_encoder(tparams, options):
    """
    Encoder only, for images
    """
    # image features
    im = tensor.matrix('im', dtype='float32')

    return [im], encode_images(tparams, options, im)


def build_errors(tparams, options):
    """ Given sentence and image embeddings, compute the error matrix """
    # input features
    s_emb = tensor.matrix('s_emb', dtype='float32')
    im_emb = tensor.matrix('im_emb', dtype='float32')

    errs = None
    if options['method'] in ['order', 'order_discriminator']:
        # trick to make Theano not optimize this into a single matrix op, and
        # overflow memory
        indices = tensor.arange(s_emb.shape[0])
        errs, _ = theano.map(lambda i, s, im: order_violations(s[i], im, options).sum(axis=1).flatten(),
                             sequences=[indices],
                             non_sequences=[s_emb, im_emb])
    else:
        errs = - tensor.dot(s_emb, im_emb.T)

    s_disc = discriminate(tparams, s_emb, options)
    im_disc = discriminate(tparams, im_emb, options)

    dcost = discriminator_loss(s_disc, im_disc, options)

    return [s_emb, im_emb], errs, dcost
