import tensorflow as tf

__all__ = [
    'eot',
    'eot_tar'
]

NOISE_SIGMA = 0.1
ENSEMBLE_SIZE = 30

def eot(model, x, eps=0.01, epochs=1, clip_min=0., clip_max=1.):

    # Compute the eot gradient
    def eot_gradient(model, x):  # The size of tensor x is [1, shape] ([1, 32, 32, 3])

        def defend(input_tensor):
            rnd = tf.random_normal(input_tensor.get_shape().as_list(), 0.0, NOISE_SIGMA, dtype=tf.float32)
            return input_tensor + rnd

        loss_fn = tf.nn.softmax_cross_entropy_with_logits

        ensemble_xs = tf.concat([defend(x) for _ in range(ENSEMBLE_SIZE)], axis=0)

        ensemble_preds, ensemble_logits = model(ensemble_xs, logits=True)
        ydim = ensemble_preds.shape[1]

        indices = tf.argmax(ensemble_preds, axis=1)

        print(ensemble_logits.shape)

        target = tf.cond(
            tf.equal(ydim, 1),
            lambda: tf.nn.relu(tf.sign(ensemble_preds - 0.5)),
            lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

        ensemble_loss = tf.reduce_mean(loss_fn(labels=target, logits=ensemble_logits))
        g = tf.gradients(ensemble_loss, x)


        return g

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        dy_dx, = eot_gradient(model, xadv)
        xadv = tf.stop_gradient(xadv + eps*tf.sign(dy_dx))

        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    def _f(xi):
        xi = tf.expand_dims(xi, axis=0)
        xi, _ = tf.while_loop(_cond, _body, (xi, 0), back_prop=False,
                                name='eot_fast_gradient')
        return xi



    eps = tf.abs(eps)
    xadv = tf.identity(x)
    xadv = tf.map_fn(_f, xadv, dtype=(tf.float32), back_prop=False,
                            name='eot_fast_gradient')

    xadv = tf.squeeze(xadv)
    #xadv = tf.expand_dims(xadv, axis=3)

    return xadv

def eot_tar(model, x, eps=0.01, epochs=1, clip_min=0., clip_max=1.):

    # Compute the eot gradient
    def eot_gradient(model, x):  # The size of tensor x is [1, shape] ([1, 32, 32, 3])

        def defend(input_tensor):
            rnd = tf.random_normal(input_tensor.get_shape().as_list(), 0.0, NOISE_SIGMA, dtype=tf.float32)
            return input_tensor + rnd

        loss_fn = tf.nn.softmax_cross_entropy_with_logits

        ensemble_xs = tf.concat([defend(x) for _ in range(ENSEMBLE_SIZE)], axis=0)

        ensemble_preds, ensemble_logits = model(ensemble_xs, logits=True)
        ydim = ensemble_preds.shape[1]

        indices = tf.argmin(ensemble_preds, axis=1)


        target = tf.cond(
            tf.equal(ydim, 1),
            lambda: 1 - ensemble_preds,
            lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

        ensemble_loss = tf.reduce_mean(loss_fn(labels=target, logits=ensemble_logits))
        g = tf.gradients(ensemble_loss, x)


        return g

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        dy_dx, = eot_gradient(model, xadv)
        xadv = tf.stop_gradient(xadv + eps*tf.sign(dy_dx))

        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    def _f(xi):
        xi = tf.expand_dims(xi, axis=0)
        xi, _ = tf.while_loop(_cond, _body, (xi, 0), back_prop=False,
                                name='eot_fast_gradient')
        return xi



    eps = -tf.abs(eps)
    xadv = tf.identity(x)
    xadv = tf.map_fn(_f, xadv, dtype=(tf.float32), back_prop=False,
                            name='eot_fast_gradient')

    xadv = tf.squeeze(xadv)
    #xadv = tf.expand_dims(xadv, axis=3)

    return xadv











