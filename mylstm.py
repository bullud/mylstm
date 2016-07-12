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

def lstm_layer(sparams, state_below, options, mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, sparams['lstm_U'])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, options['hidden_dim']))
        f = T.nnet.sigmoid(_slice(preact, 1, options['hidden_dim']))
        o = T.nnet.sigmoid(_slice(preact, 2, options['hidden_dim']))
        c = T.tanh(_slice(preact, 3, options['hidden_dim']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, sparams['lstm_W']) +
                   sparams['lstm_b'])

    hidden_dim = options['hidden_dim']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           hidden_dim),
                                              T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           hidden_dim)],
                                name='lstm_layers',
                                n_steps=nsteps)
    return rval[0]

def build_model(options):
    # define switch for dropout, use or not use
    use_dropout = theano.shared(numpy_floatX(0.))

    #define x, y and mask of a batch
    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=config.floatX)
    y = T.vector('y', dtype='int64')

    #define model params
    params = init_params(options)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    #x:
    #s1_w1,  s2_w1,  s3_w1, ...
    #s1_w2,  s2_w2,  s3_w2, ...
    #s1_w3,  s2_w3,  s3_w3, ...
    #s1_w4,  s2_w4,  s3_w4, ...
    #...

    #tparams['Wemb'][x.flatten()] - >
    #[[s1_w1_v1, s1_w1_v2, s1_w1_v3,..., s2_w1_v1, s2_w1_v2, s2_w1_v3, s3_w1_v1, s3_w1_v2, s3_w1_v3, ..., ]
    emb = params['Wemb'][x.flatten()].reshape([n_timesteps,
                                               n_samples,
                                               options['hidden_dim']])

    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
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
