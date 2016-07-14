import theano
import theano.tensor as T
from theano import config
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
import imdb

SEED = 123
np.random.seed(SEED)

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
    params['b'] = np.zeros((options['ydim'],)).astype(config.floatX) #row vector

    return make_shared(params)

def dropout_layer(state_before, use_noise, trng):
    hidden = T.switch(use_noise,
                      (state_before *
                       trng.binomial(state_before.shape,
                                   p=0.5, n=1,
                                   dtype=state_before.dtype)),
                       state_before * 0.5)
    return hidden

def lstm_layer(sparams, state_below, options, mask=None):
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

        c = i * c + f * c_
        #m_[:, None] change row to col
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    state_below = (T.dot(state_below, sparams['lstm_W']) + sparams['lstm_b'])

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

def build_model(params, options):
    # define switch for dropout, use or not use
    use_noise = theano.shared(numpy_floatX(0.))

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

    proj = lstm_layer(params, emb, options, mask=mask)


    #compute mean pooling
    proj = (proj * mask[:, :, None]).sum(axis=0)
    proj = proj / mask.sum(axis=0)[:, None]

    trng = RandomStreams(SEED)
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    #compute predited probalility
    pred = T.nnet.softmax(T.dot(proj, params['U']) + params['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    #define cost function
    cost = -T.log(pred[T.arange(n_samples), y] + off).mean()

    print("params", list(params.values()))

    grads = T.grad(cost, wrt=list(params.values()))

    print("end")
    return use_noise, x, mask, y, f_pred_prob, f_pred, cost

def train_model(
    hidden_dim=128, #the dim of the hidden unit, and the dim of the word embeding
    maxlen=100,     #the max words of the sentence, sentence longer than this get ignored
    batch_size=16,  # The batch size during training
    lrate=0.0001,   # Learning rate for sgd (not used for adadelta and rmsprop)
    decay_c=0.,     # Weight decay for the classifier applied to the U weights.
    max_epochs=100, # The maximum number of epoch to run
    n_words=10000,  # Vocabulary size, The number of word to keep in the vocabulary.All extra words are set to unknow (1).
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
):
    model_options = locals().copy()
    print("model_paras", model_options)

    train, valid, test = imdb.load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    ydim = np.max(train[1]) + 1  #dim of the output, 0 negative 1 positive
    model_options['ydim'] = ydim

    print('Building model')
    s_params = init_params(model_options)

    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(s_params, model_options)

    #if decay_c > 0.:
    #    decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
    #    weight_decay = (s_params['U'] ** 2).sum()
    #    weight_decay *= decay_c
    #    cost += weight_decay

    print("params1", list(s_params.values()))

    #grads = T.grad(cost, wrt=list(s_params.values()))
    grads = T.grad(cost, wrt=[s_params['Wemb']])
    #print(list(s_params.values()))

    #grads = T.grad(cost, wrt=list(s_params.values()))
    #f_grad = theano.function([x, mask, y], grads, name='f_grad')


    return None


if __name__ == "__main__":
    train_model()
