import numpy as np
import tensorflow as tf


class DataProvider:
    """
    Parameters
    ----------
        data_container: DataContainer
            Contains the dataset.
        ntrain: int
            Number of samples in the training set.
        nval: int
            Number of samples in the validation set.
        batch_size: int
            Number of samples to process at once.
        seed: int
            Seed for drawing samples into train and val set (and shuffle).
        random_split: bool
            If True put the samples randomly into the subsets else in order.
        shuffle: bool
            If True shuffle the samples after each epoch.
        sample_with_replacement: bool
            Sample data from the dataset with replacement.
    """

    def __init__(
        self,
        data_container,
        ntrain: int,
        nval: int,
        batch_size: int = 1,
        seed: int = None,
        random_split: bool = False,
        shuffle: bool = True,
        sample_with_replacement: bool = False,
    ):
        self.data_container = data_container
        self._ndata = len(data_container)
        self.nsamples = {
            "train": ntrain,
            "val": nval,
            "test": len(data_container) - ntrain - nval,
        }
        self.batch_size = batch_size
        self.random_split = random_split
        self.shuffle = shuffle

        # Random state parameter, such that random operations are reproducible if wanted
        self._random_state = np.random.RandomState(seed=seed)

        all_idx = np.arange(len(self.data_container))
        if random_split:
            # Shuffle indices
            all_idx = self._random_state.permutation(all_idx)

        if sample_with_replacement:
            # Sample with replacement so as to train an ensemble of Dimenets
            all_idx = self._random_state.choice(all_idx, self._ndata, replace=True)

        # Store indices of training, validation and test data
        self.idx = {
            "train": all_idx[0:ntrain],
            "val": all_idx[ntrain : ntrain + nval],
            "test": all_idx[ntrain + nval :],
        }

        # Index for retrieving batches
        self.idx_in_epoch = {"train": 0, "val": 0, "test": 0}

        self.dtypes = self.data_container.get_dtypes()
        self.shapes = self.data_container.get_shapes()

    def shuffle_train(self):
        """Shuffle the training data"""
        self.idx["train"] = self._random_state.permutation(self.idx["train"])

    def get_batch(self, split):
        """Return a batch of samples from the training set"""
        start = self.idx_in_epoch[split]

        # Is epoch finished?
        if self.idx_in_epoch[split] == self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0

        # shuffle training set at start of epoch
        if start == 0 and split == "train" and self.shuffle is True:
            self.shuffle_train()

        # Set end of batch
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]

        samples = self.data_container[self.idx[split][start:end]]

        return samples

    def get_dataset(self, split, batch_size=None):
        """Return a tf.data.Dataset generator.

        Returns
        -------
            dataset: Iterator
                Dataset that returns batches of data for training and validation.
        """
        if batch_size is not None:
            self.batch_size = batch_size

        def generator():
            while True:
                batch = self.get_batch(split)
                inputs = {}
                for key in self.dtypes[0]:
                    inputs[key] = batch[key]
                targets = {}
                for key in self.dtypes[1]:
                    targets[key] = batch[key]
                yield (inputs, targets)

        return iter(
            tf.data.Dataset.from_generator(
                generator, output_types=self.dtypes, output_shapes=self.shapes
            ).prefetch(tf.data.experimental.AUTOTUNE)
        )
