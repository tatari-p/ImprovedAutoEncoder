import tensorflow as tf
from tensorflow.contrib import layers
from ResNET.model import *


class ImprovedAutoEncoder:

    def __init__(self, name):
        self.name = name

        self.x = None
        self.cor = None
        self.z_in = None
        self.training = None
        self.carry = None
        self.g_lr = None
        self.d_lr = None
        self.rec_lr = None
        self.z = None
        self.rec = None
        self.dec = None
        self.alpha = None
        self.rec_loss = None
        self.d_loss = None
        self.g_loss = None
        self.train_op = None

    def build(self, img_size, num_z, num_units, num_repeats, batch_norm=True, gray_scale=False):

        with tf.variable_scope(self.name):
            num_inputs = 1 if gray_scale else 3
            self.x = tf.placeholder(tf.float32, [None, *img_size, num_inputs])
            self.cor = tf.placeholder(tf.float32, [None, *img_size, num_inputs])
            cor_x = self.x+self.cor
            self.training = tf.placeholder(tf.bool, shape=())
            self.carry = tf.placeholder(tf.float32, shape=())
            self.g_lr = tf.placeholder(tf.float32, shape=())
            self.d_lr = tf.placeholder(tf.float32, shape=())
            self.rec_lr = tf.placeholder(tf.float32, shape=())
            self.z = encoder("encoder", cor_x, num_z, num_units, num_repeats, self.training, carry=self.carry, batch_norm=batch_norm)
            self.rec = decoder("decoder", self.z, num_units, num_repeats, self.training, carry=self.carry, batch_norm=batch_norm)
            self.z_in = tf.placeholder(tf.float32, [None, num_z])
            self.dec = decoder("decoder", self.z_in, num_units, num_repeats, self.training, carry=self.carry, batch_norm=batch_norm, reuse=True)
            self.alpha = tf.placeholder(tf.float32, shape=())

            d_real = create_resnet_18("discriminator", self.x, self.training, 1)
            d_fake = create_resnet_18("discriminator", self.dec, self.training, 1, reuse=True)

            d_loss_real = tf.reduce_mean(tf.square(d_real-1))
            d_loss_fake = tf.reduce_mean(tf.square(d_fake))

            g_loss_fake = tf.reduce_mean(tf.square(d_fake-1))

            self.rec_loss = self.alpha*tf.reduce_mean(tf.square(self.rec-self.x))+0.01*tf.reduce_mean(tf.square(self.z))

            self.g_loss = g_loss_fake
            self.d_loss = d_loss_real+d_loss_fake

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            t_vars = tf.trainable_variables()
            d_vars = [t_var for t_var in t_vars if "discriminator" in t_var.name]
            g_vars = [t_var for t_var in t_vars if "encoder" in t_var.name or "decoder" in t_var.name]

            with tf.control_dependencies(update_ops):
                d_train_op = tf.train.AdamOptimizer(self.d_lr).minimize(self.d_loss, var_list=d_vars)
                g_train_op = tf.train.AdamOptimizer(self.g_lr).minimize(self.g_loss, var_list=g_vars)
                rec_train_op = tf.train.AdamOptimizer(self.rec_lr).minimize(self.rec_loss, var_list=g_vars)

            self.train_op = [rec_train_op, d_train_op, g_train_op]

            tf.summary.scalar("d_loss", self.d_loss)
            tf.summary.scalar("g_loss", self.g_loss)
            tf.summary.scalar("rec_loss", self.rec_loss)
            tf.summary.histogram("z", self.z)
            tf.summary.histogram("z_in", self.z_in)


def decoder(name, z, num_units, num_repeats, is_training, carry=None, batch_norm=False, unit=8, reuse=False):

    with tf.variable_scope(name, reuse=reuse):
        batch_norm_fn = tf.layers.batch_normalization if batch_norm else None
        normalize_params = {"momentum": 0.9, "training": is_training, "scale": True, "fused": True} if batch_norm else None
        initializer = tf.random_normal_initializer(0., 0.02)
        num_output = unit*unit*num_units
        img = layers.fully_connected(z, num_output, weights_initializer=initializer, activation_fn=None, normalizer_fn=batch_norm_fn, normalizer_params=normalize_params)
        img = inj = tf.reshape(img, [-1, unit, unit, num_units])

        for i in range(num_repeats):
            img = in_x = layers.conv2d(img, num_units, 3, 1, weights_initializer=initializer, activation_fn=tf.nn.elu)
            img = layers.conv2d(img, num_units, 3, 1, weights_initializer=initializer, biases_initializer=None, activation_fn=tf.nn.elu, normalizer_fn=batch_norm_fn, normalizer_params=normalize_params)

            if carry is not None:
                img = carry*in_x+(1-carry)*img

            if i < num_repeats-1:
                img_shape = img.get_shape()
                inj = tf.image.resize_nearest_neighbor(inj, (2*int(img_shape[1]), 2*int(img_shape[2])))
                img = tf.image.resize_nearest_neighbor(img, (2*int(img_shape[1]), 2*int(img_shape[2])))
                img = tf.concat([img, inj], axis=-1)

        out = layers.conv2d(img, 3, 3, 1, weights_initializer=initializer, activation_fn=None)

    return out


def encoder(name, img, num_z, num_units, num_repeats, is_training, carry=None, batch_norm=False, unit=8, reuse=False):

    with tf.variable_scope(name, reuse=reuse):
        batch_norm_fn = tf.layers.batch_normalization if batch_norm else None
        normalize_params = {"momentum": 0.9, "training": is_training, "scale": True, "fused": True} if batch_norm else None
        initializer = tf.random_normal_initializer(0., 0.02)
        img = layers.conv2d(img, num_units, 3, 1, weights_initializer=initializer, biases_initializer=None, activation_fn=tf.nn.elu, normalizer_fn=batch_norm_fn, normalizer_params=normalize_params)

        for i in range(num_repeats):
            img = in_x = layers.conv2d(img, num_units*(i+1), 3, 1, weights_initializer=initializer, activation_fn=tf.nn.elu)
            img = layers.conv2d(img, num_units*(i+1), 3, 1, weights_initializer=initializer, biases_initializer=None, activation_fn=tf.nn.elu, normalizer_fn=batch_norm_fn, normalizer_params=normalize_params)

            if carry is not None:
                img = carry*in_x+(1-carry)*img

            if i < num_repeats-1:
                img = layers.conv2d(img, num_units*(i+1), 3, 2, weights_initializer=initializer, activation_fn=tf.nn.elu)

        img = tf.reshape(img, [-1, unit*unit*num_units*num_repeats])
        z = layers.fully_connected(img, num_z, weights_initializer=initializer, activation_fn=None)

    return z