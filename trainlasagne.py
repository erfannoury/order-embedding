"""
Main trainer function
"""
import theano
import lasagne
import cPickle as pkl
import json

import os
import time

import datasource

from utils import *
from optim import adam
from model import init_params, build_errors, contrastive_loss, discriminator_loss
from vocab import build_dictionary
from evaluation import t2i, i2t
from tools import encode_sentences, encode_images, compute_errors
from datasets import load_dataset


# main trainer
def lasagnetrainer(load_from=None,
                   save_dir='snapshots',
                   name='anon',
                   **kwargs):
    """
    :param load_from: location to load parameters + options from
    :param name: name of model, used as location to save parameters + options
    """

    curr_model = dict()

    # load old model, including parameters, but overwrite with new options
    if load_from:
        print 'reloading...' + load_from
        with open('%s.pkl' % load_from, 'rb') as f:
            curr_model = pkl.load(f)
    else:
        curr_model['options'] = {}

    for k, v in kwargs.iteritems():
        curr_model['options'][k] = v

    model_options = curr_model['options']

    # initialize logger
    import datetime
    timestampedName = datetime.datetime.now().strftime(
        '%Y_%m_%d_%H_%M_%S') + '_' + name

    from logger import Log
    log = Log(name=timestampedName, hyperparams=model_options, saveDir='vis/training',
              xLabel='Examples Seen', saveFrequency=1)

    print curr_model['options']

    # Load training and development sets
    print 'Loading dataset'
    dataset = load_dataset(model_options['data'], cnn=model_options[
                           'cnn'], load_train=True)
    train = dataset['train']
    dev = dataset['dev']

    # Create dictionary
    print 'Creating dictionary'
    worddict = build_dictionary(train['caps'] + dev['caps'])
    print 'Dictionary size: ' + str(len(worddict))
    curr_model['worddict'] = worddict
    curr_model['options']['n_words'] = len(worddict) + 2

    # save model
    pkl.dump(curr_model, open('%s/%s.pkl' % (save_dir, name), 'wb'))

    print 'Loading data'
    train_iter = datasource.Datasource(train, batch_size=model_options[
                                       'batch_size'], worddict=worddict)
    dev = datasource.Datasource(dev, worddict=worddict)
    dev_caps, dev_ims = dev.all()

    print 'Building model'
    ###########################################
    # params = init_params(model_options)
    ###########################################
    # reload parameters
    if load_from is not None and os.path.exists(load_from):
        # params = load_params(load_from, params)
        params = pkl.load(load_from)

    ###########################################
    # tparams = init_tparams(params)
    ###########################################

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='int8')
    im = tensor.matrix('im', dtype='float32')

    unif_gate = lasagne.layers.Gate(
        W_cell=None, W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1))

    l_in_sent = lasagne.layers.InputLayer((None, None), input_var=x.T)
    l_in_im = lasagne.layers.InputLayer(
        (None, model_options['dim_image']), input_var=im)
    l_in_mask = lasagne.layers.InputLayer((None, None), input_var=mask.T)

    l_emb = lasagne.layers.EmbeddingLayer(
        l_in_sent, input_size=model_options['n_words'], output_size=model_options['dim_word'])
    l_gru = lasagne.layers.GRULayer(
        l_emb, model_options['dim'],
        resetgate=lasagne.layers.Gate(
            W_cell=None,
            W_in=lasagne.init.Uniform(0.1),
            W_hid=lasagne.init.Uniform(0.1)),
        updategate=lasagne.layers.Gate(
            W_cell=None,
            W_in=lasagne.init.Orthogonal(),
            W_hid=lasagne.init.Orthogonal()),
        hidden_update=lasagne.layers.Gate(
            W_cell=None,
            W_in=lasagne.init.Uniform(0.1),
            W_hid=lasagne.init.Uniform(0.1),
            nonlinearity=lasagne.nonlinearities.tanh),
        mask_input=l_in_mask,
        only_return_final=True
    )

    l_im_emb = lasagne.layers.DenseLayer(l_in_im, model_options['dim'],
                                         nonlinearity=lasagne.nonlinearities.linear
                                         )

    l_disc_in = lasagne.layers.InputLayer((None, model_options['dim']))
    l_disc_hid = lasagne.layers.DenseLayer(l_disc_in, model_options['disc_hid_size'],
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           b=lasagne.init.Constant(1.0)
                                           )
    l_disc = lasagne.layers.DenseLayer(l_disc_hid, 1,
                                       nonlinearity=lasagne.nonlinearities.sigmoid
                                       )

    s = lasagne.layers.get_output(l_gru)
    s_emb = l2norm(s)
    if model_options['abs']:
        s_emb = abs(s)

    im_emb = lasagne.layers.get_output(l_im_emb)
    im_emb = l2norm(im_emb)
    if model_options['abs']:
        im_emb = abs(im_emb)

    s_disc = lasagne.layers.get_output(l_disc,
                                       inputs=lasagne.layers.get_output(l_gru)
                                       )

    im_disc = lasagne.layers.get_output(l_disc,
                                        inputs=lasagne.layers.get_output(
                                            l_im_emb)
                                        )

    ccost = contrastive_loss(s_emb, im_emb, model_options)
    dcost = discriminator_loss(s_disc, im_disc, model_options)
    cost = ccost + dcost

    #############################################################
    # inps, cost, ext_costs = build_model(tparams, model_options)
    #############################################################

    ######################################################################
    # print 'Building sentence encoder'
    # inps_se, sentences = build_sentence_encoder(tparams, model_options)
    # f_senc = theano.function(inps_se, sentences, profile=False)
    ######################################################################
    f_senc = theano.function([x, mask], s_emb)

    ###############################################################
    # print 'Building image encoder'
    # inps_ie, images = build_image_encoder(tparams, model_options)
    # f_ienc = theano.function(inps_ie, images, profile=False)
    ###############################################################
    f_ienc = theano.function([im], im_emb)

    ###################################################
    # print 'Building f_grad...',
    # grads = tensor.grad(cost, wrt=itemlist(tparams))
    ###################################################
    im_emb_params = lasagne.layers.get_all_params(l_im_emb)
    s_emb_params = lasagne.layers.get_all_params(l_gru)
    disc_params = lasagne.layers.get_all_params(l_disc)
    allparams = im_emb_params + s_emb_params + disc_params
    grads = tensor.grad(cost, wrt=allparams)

    print 'Building errors..'
    ##############################################################
    # inps_err, errs, dcost = build_errors(tparams, model_options)
    # f_err = theano.function(inps_err, errs, profile=False)
    # f_dcost = theano.function(inps_err, dcost, profile=False)
    ##############################################################
    f_err = theano.function([x, mask, im], cost, profile=False)
    f_dcost = theano.function([x, mask, im], dcost, profile=False)
    f_ccost = theano.function([x, mask, im], ccost, profile=False)

    curr_model['f_senc'] = f_senc
    curr_model['f_ienc'] = f_ienc
    curr_model['f_err'] = f_err

    if model_options['grad_clip'] > 0.:
        grads = [maxnorm(g, model_options['grad_clip']) for g in grads]

    lr = theano.shared(numpy.float32(model_options['lrate']), name='lr')
    print 'Building optimizers...',
    #############################################################
    # (compute gradients), (updates parameters)
    # f_grad_shared, f_update = eval(model_options['optimizer'])(
    # lr, tparams, grads, inps, cost)
    #############################################################
    adam_update = lasagne.updates.adam(grads, allparams, lr, 0.1, 0.001)
    f_update = theano.function([x, mask, im], updates=adam_update)

    print 'Optimization'

    uidx = 0
    curr = 0
    n_samples = 0

    for eidx in xrange(model_options['max_epochs']):

        print 'Epoch ', eidx

        for x, mask, im in train_iter:
            mask = mask.astype(numpy.int8)
            n_samples += x.shape[1]
            uidx += 1

            # Update
            ud_start = time.time()
            ####################################
            # cost = f_grad_shared(x, mask, im)
            # f_update(model_options['lrate'])
            ####################################
            print 'x.shape: {}, mask.shape: {}, im.shape: {}'.format(x.shape, mask.shape, im.shape)
            cost = f_err(x, mask, im)
            f_update(x, mask, im)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, model_options['dispFreq']) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud
                log.update({'Error': float(cost)}, n_samples)

            if numpy.mod(uidx, model_options['validFreq']) == 0:

                print 'Computing results...'

                # encode sentences efficiently
                dev_s = encode_sentences(
                    curr_model, dev_caps, batch_size=model_options['batch_size'])
                dev_i = encode_images(curr_model, dev_ims)

                # compute errors
                dev_errs = compute_errors(curr_model, dev_s, dev_i)
                dcost_errs = f_dcost(dev_s, dev_i)

                # compute ranking error
                (r1, r5, r10, medr, meanr), vis_details = t2i(
                    dev_errs, vis_details=True)
                (r1i, r5i, r10i, medri, meanri) = i2t(dev_errs)
                print "Discriminator cost: %.3f" % (dcost_errs)
                print "Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr)
                log.update({'R@1': r1, 'R@5': r5, 'R@10': r10,
                            'median_rank': medr, 'mean_rank': meanr}, n_samples)
                print "Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanri)
                log.update({'Image2Caption_R@1': r1i, 'Image2Caption_R@5': r5i, 'Image2CaptionR@10': r10i,
                            'Image2Caption_median_rank': medri, 'Image2Caption_mean_rank': meanri}, n_samples)

                tot = r1 + r5 + r10
                if tot > curr:
                    curr = tot
                    # Save parameters
                    print 'Saving...',
                    numpy.savez('%s/%s' % (save_dir, name), **unzip(tparams))
                    print 'Done'
                    vis_details['hyperparams'] = model_options
                    # Save visualization details
                    # TODO Fix this:
                    #   TypeError, 942 is not JSON serializable
                    # with open('vis/roc/%s/%s.json' % (model_options['data'], timestampedName), 'w') as f:
                    # json.dump(vis_details, f)
                    # Add the new model to the index
            try:
                index = json.load(open('vis/roc/index.json', 'r'))
            except IOError:
                index = {model_options['data']: []}

                models = index[model_options['data']]
                if timestampedName not in models:
                    models.append(timestampedName)

                with open('vis/roc/index.json', 'w') as f:
                    json.dump(index, f)

        print 'Seen %d samples' % n_samples

if __name__ == '__main__':
    pass
