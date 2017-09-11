import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import tensorflow as tf
layers = tf.layers
# from scipy.stats import norm


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--sample_num', type=int, default=100)
parser.add_argument('--data_dim', type=int, default=5)
parser.add_argument('--z_dim', type=int, default=50)
parser.add_argument('--width', type=int, default=5,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=5,
                    help='num of generator layers')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adagrad', 'adam', 'gradientdescent'])

args = parser.parse_args()
sample_num = args.sample_num
data_dim = args.data_dim
z_dim = args.z_dim
width = args.width
depth = args.depth
learning_rate = args.learning_rate
optimizer = args.optimizer
save_tag = 'dn{}_dim{}_zd{}_w{}_d{}_lr{}_op_{}'.format(sample_num, data_dim,
                                                       z_dim, width, depth,
                                                       learning_rate,
                                                       optimizer)
out_dim = data_dim
activation = tf.nn.elu
total_num_runs = 400101

# Set up true, training data.
UNNORMED_DATA = np.load('mcmc_samples.npy')
num_data = len(UNNORMED_DATA)
var_maxs = UNNORMED_DATA.max(axis=0)
DATA = UNNORMED_DATA / var_maxs
# DATA = DATA[:, [3, 1, 4, 2, 0]]
DATA_names = ['intercept', 'age', 'age2', 'educ', 'hours']


def get_random_z(gen_num, z_dim):
    """Generates 2d array of noise input data."""
    return np.random.uniform(size=[gen_num, z_dim],
                             low=-1.0, high=1.0)


# Define generator.
def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=1,
              reuse=False):
    """Generates output, given noise input."""
    with tf.variable_scope('generator', reuse=reuse):
        x = layers.dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        out = layers.dense(x, out_dim, activation=None)
    return out


# Build model.
x = tf.placeholder(tf.float64, [sample_num, data_dim], name='x')
z = tf.placeholder(tf.float64, [sample_num, z_dim], name='z')
g = generator(z, width=width, depth=depth, activation=activation,
              out_dim=out_dim)
v = tf.concat([x, g], 0)
VVT = tf.matmul(v, tf.transpose(v))
sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
sigma = 1
K = tf.exp(-0.5 * (1 / sigma) * exp_object)
K_xx = K[:sample_num, :sample_num]
K_yy = K[sample_num:, sample_num:]
K_xy = K[:sample_num, sample_num:]
K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
num_combos = sample_num * (sample_num - 1) / 2
mmd = (tf.reduce_sum(K_xx_upper) / num_combos +
       tf.reduce_sum(K_yy_upper) / num_combos -
       2 * tf.reduce_sum(K_xy) / (sample_num * sample_num))
g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
if optimizer == 'adagrad':
    opt = tf.train.AdagradOptimizer
elif optimizer == 'adam':
    opt = tf.train.AdamOptimizer
else:
    opt = tf.train.GradientDescentOptimizer
g_optim = opt(learning_rate).minimize(mmd, var_list=g_vars)

# Train.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print args
start_time = time()
for i in range(total_num_runs):
    x_resampled = DATA[np.random.choice(DATA.shape[0], sample_num), :]
    sess.run(g_optim,
             feed_dict={
                 z: get_random_z(sample_num, z_dim),
                 x: x_resampled})

    if i % 2000 == 100:
        mmd_out, g_out = sess.run(
            [mmd, g],
            feed_dict={
                z: get_random_z(sample_num, z_dim),
                x: x_resampled})
        print '\niter:{} mmd = {}'.format(i, mmd_out)

        # Compare distribution of each variable on data and gen.
        fig, ax = plt.subplots(5, 1, figsize=(8, 15))
        for var_idx in range(data_dim):
            print 'Var: {}'.format(var_idx)
            betas_data = (np.array([j[var_idx] for j in DATA]) *
                          var_maxs[var_idx])
            betas_gen = (np.array([j[var_idx] for j in g_out]) *
                         var_maxs[var_idx])

            # Print summary of distributions.
            print ('  DATA: min={:.2f}, p25={:.2f}, p50={:.2f}, p75={:.2f}, '
                   'max={:.2f}').format(
                min(betas_data), np.percentile(betas_data, 25),
                np.percentile(betas_data, 50), np.percentile(betas_data, 75),
                max(betas_data))
            print ('  GEN: min={:.2f}, p25={:.2f}, p50={:.2f}, p75={:.2f}, '
                   'max={:.2f}').format(
                min(betas_gen), np.percentile(betas_gen, 25),
                np.percentile(betas_gen, 50), np.percentile(betas_gen, 75),
                max(betas_gen))

            # Plot summary of distributions.
            ax[var_idx].plot(np.sort(betas_data),
                             np.array(range(num_data))/float(num_data),
                             label='data', color='g', alpha=0.3)
            ax[var_idx].plot(np.sort(betas_gen),
                             np.array(range(sample_num))/float(sample_num),
                             label='gen', color='b', alpha=0.3)
            ax[var_idx].set_title(DATA_names[var_idx])
            ax[var_idx].legend()
            ax[var_idx].set_xlim([min(betas_data), max(betas_data)])

        plt.suptitle('GAN-learned posterior distributions: iter{}'.format(i))
        plt.savefig('var_cdfs_i{}'.format(i))

        if i > 0:
            elapsed_time = time() - start_time
            time_per_iter = elapsed_time / i
            total_est = elapsed_time / i * total_num_runs
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            print ('\nTime (s). Elapsed: {:.2f}, Avg/iter: {:.4f},'
                   ' Total est.: {}').format(elapsed_time, time_per_iter,
                                             total_est_str)
