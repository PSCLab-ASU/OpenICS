# Get input

from tensorflow.examples.tutorials.mnist import input_data

def mnist_data_iteratior():
    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    def iterator(hparams, num_batches):
        for _ in range(num_batches):
            yield mnist.train.next_batch(hparams.batch_size)
    return iterator
