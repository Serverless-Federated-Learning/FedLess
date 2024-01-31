import functools
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PartitionDataset:
    def __init__(self, labels, original_targets, indices, sizes, type, alpha=100, random_seed=42):

        np.random.seed(random_seed)
        self.sizes = sizes
        self.type = type
        self.alpha = alpha
        self.original_targets = original_targets

        self.data_size = len(labels)
        self.labels = labels
        self.indices = indices

        self.partitions = []

        self.create_partitioned_indices()

    def create_partitioned_indices(self):

        if self.type == "random":
            distributed_indices = sorted(self.indices, key=lambda k: np.random.random())

        elif self.type == "origin":
            distributed_indices = self.indices

        elif self.type == "non_iid_dirichlet":
            n_workers = len(self.sizes)

            list_of_indices = build_non_iid_by_dirichlet(
                indices_targets=np.array([(idx, label) for idx, label in zip(self.indices, self.labels)]),
                alpha=self.alpha,
                n_workers=n_workers,
                classes=np.unique(self.labels).tolist(),
            )
            distributed_indices = functools.reduce(lambda a, b: a + b, list_of_indices)
            # distributed_indices =
        else:
            raise NotImplementedError(f"The partition scheme={self.type} is not implemented yet")

        # partition indices.
        from_index = 0
        for partition_size in self.sizes:
            to_index = from_index + int(partition_size * self.data_size)
            self.partitions.append(distributed_indices[from_index:to_index])
            from_index = to_index

        store_class_distribution(self.partitions, self.original_targets, self.alpha)

    def get_partitioned_indices(self):
        return self.partitions


def build_non_iid_by_dirichlet(indices_targets, alpha, classes, n_workers):

    # random shuffle targets indices.
    np.random.shuffle(indices_targets)
    total_indices = len(indices_targets)
    min_size = 0
    net_dataidx_map = []

    total_trials = 10000
    current_trial = 0
    while min_size < int(0.50 * total_indices / n_workers) and current_trial <= total_trials:
        idx_batch = [[] for _ in range(n_workers)]

        for _class in classes:
            idx_class = np.where(indices_targets[:, 1] == _class)[0]
            idx_class = indices_targets[idx_class, 0]

            np.random.shuffle(idx_class)

            proportions = np.random.dirichlet(np.repeat(alpha, n_workers))

            # Balance
            proportions = np.array(
                [p * (len(idx_j) < total_indices / n_workers) for p, idx_j in zip(proportions, idx_batch)]
            )

            proportions = proportions / proportions.sum()

            proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_class, proportions))]

            min_size = min([len(idx_j) for idx_j in idx_batch])

            current_trial = current_trial + 1

    # Sampling for incomplete client distributions
    if current_trial > total_trials + 2:
        for idx_main, batch_main in enumerate(idx_batch):
            if len(batch_main) < int(0.50 * total_indices / n_workers):
                # Sample from other client distributions
                for idx_curr, batch_other in enumerate(idx_batch):
                    if idx_main != idx_curr and len(batch_other) >= int(total_indices / n_workers):
                        for item_idx in range(int(0.50 * total_indices / n_workers)):
                            batch_main.append(batch_other.pop(item_idx))

    # Making a check if all clients have elements
    for batch in idx_batch:
        assert len(batch) != 0, "Some clients have zero elements after Non-IID distribution!"

    for j in range(n_workers):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map.append(idx_batch[j])

    return net_dataidx_map


def store_class_distribution(partitions, targets, alpha):
    targets_of_partitions = {}
    distribution_df = pd.DataFrame(columns=["client", "class", "n_samples"])
    targets_np = np.array(targets)
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(targets_np[partition], return_counts=True)
        targets_of_partitions[idx] = list(zip(unique_elements, counts_elements))
    logger.info("Class Distribution after data partitioning")
    for key, val in targets_of_partitions.items():
        for client_tuple in val:
            distribution_df = distribution_df.append(
                {"client": key, "class": client_tuple[0], "n_samples": client_tuple[1]}, ignore_index=True
            )

    logger.info(distribution_df)
    distribution_df.to_csv(f"/home/ubuntu/Enabling-KD-with-FedLess/out/distribution_alpha_{alpha}.csv")
    return targets_of_partitions
