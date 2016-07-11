import theano
import theano.tensor as T
from theano import config
import numpy as np

from collections import OrderedDict
import imdb

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def make_shared(params):
    sparams = OrderedDict()
    for kk, pp in params.items():
        sparams[kk] = theano.shared(params[kk], name=kk)
    return sparams

def init_lstm_param(options, params):
    """
    Init the LSTM parameter:
    :see: init_params
    """
    hidden_dim = options['hidden_dim']
    W = np.concatenate([ortho_weight(hidden_dim),
                        ortho_weight(hidden_dim),
                        ortho_weight(hidden_dim),
                        ortho_weight(hidden_dim)], axis=1)
    params['lstm_W'] = W

    U = np.concatenate([ortho_weight(hidden_dim),
                        ortho_weight(hidden_dim),
                        ortho_weight(hidden_dim),
                        ortho_weight(hidden_dim)], axis=1)
    params['lstm_U'] = U

    b = np.zeros((4 * hidden_dim,))
    params['lstm_b'] = b.astype(config.floatX)

    return params

def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = np.random.rand(options['n_words'],
                           options['hidden_dim'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)

    # lstm
    params = init_lstm_param(options, params)

    # classifier
    params['U'] = 0.01 * np.random.randn(options['hidden_dim'],
                                         options['ydim']).astype(config.floatX)
    params['b'] = np.zeros((options['ydim'],)).astype(config.floatX)

    return make_shared(params)



def build_model(options):

    # define switch for dropout, use or not use
    use_dropout = theano.shared(numpy_floatX(0.))

    #define x, y and mask of a batch
    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=config.floatX)
    y = T.vector('y', dtype='int64')

    #define model params
    params = init_params(options)



    return None

def train_model(
    hidden_dim=128, #the dim of the hidden unit, and the dim of the word embeding
    maxlen=100,     #the max words of the sentence, sentence longer than this get ignored
    batch_size=16,  # The batch size during training
    lrate=0.0001,   # Learning rate for sgd (not used for adadelta and rmsprop)
    decay_c=0.,     # Weight decay for the classifier applied to the U weights.
    max_epochs=100, # The maximum number of epoch to run
    n_words=10000,  # Vocabulary size, The number of word to keep in the vocabulary.All extra words are set to unknow (1).
):
    model_options = locals().copy()
    print("model_paras", model_options)

    train, valid, test = imdb.load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    ydim = np.max(train[1]) + 1  #dim of the output, 0 negative 1 positive
    model_options['ydim'] = ydim

    print('Building model')
    s_params = init_params(model_options)

    build_model()


    return None


if __name__ == "__main__":
    train_model()
