import trainlasagne
orderdisc = {
    'method': 'order_discriminator',
    'margin': 0.05,
    'abs': True,
    'disc_coeff': 1.0,
    'disc_hid_size': 256 
}

order = {
    'method': 'order',
    'margin': 0.05,
    'abs': True,
}

symmetric = {
    'method': 'cosine',
    'margin': 0.2,
    'abs': False,
}

default_params = {
    'max_epochs': 100,
    'data': 'coco',
    'cnn': '10crop',
    'dim_image': 4096,
    'encoder': 'gru',
    'dispFreq': 10,
    'grad_clip': 2.,
    'optimizer': 'adam',
    'batch_size': 128,
    'dim': 1024,
    'dim_word': 300,
    'lrate': 0.001,
    'validFreq': 300
}


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('model', choices=['order', 'symmetric', 'orderdisc'])
# args = parser.parse_args()
# model_params = eval(args.model)
model_params = orderdisc

model_params.update(default_params)

name = model_params['method']
trainlasagne.lasagnetrainer(name=name, **model_params)
# train.trainer(name=name, **model_params)
