#!/usr/bin/python

import sys
import pathlib
import numpy as np
import tqdm


rng = np.random.default_rng()


def truncated_dense_median(predictions, n=1000, random_choice=False):
    
    if not random_choice:
        p_truncated = predictions[:n, :]
    else:
        p_truncated = rng.choice(predictions, n, axis=0, replace=False)

    print("Shape truncated:", p_truncated.shape)
    # print("0:", p_truncated[0])

    dmatrix = np.abs(p_truncated[None, :, :] - p_truncated[:, None, :])
    dmatrix = np.sum(dmatrix, axis=-1) * 0.5

    dmatrix = dmatrix.reshape([-1])

    print("Min:", np.min(dmatrix))
    print("Max:", np.max(dmatrix))

    m = np.median(dmatrix)
    return m


def sampled_median(predictions, n_samples=1000):

    p_truncated = rng.choice(predictions, n_samples * 2, axis=0, replace=True)

    # print(p_truncated.shape)

    d = np.abs(p_truncated[:n_samples] - p_truncated[n_samples:])
    d = np.sum(d, axis=-1)

    print("Min:", np.min(d))
    print("Max:", np.max(d))

    return np.median(d) / 2.0


def main():

    folder = sys.argv[1]
    print(folder)
    model = sys.argv[2]
    print(model)

    predictions_file = pathlib.Path(__file__).absolute().parent.parent / "data" / folder / "{}.bin".format(model)
    if not predictions_file.is_file():
        raise FileNotFoundError("Predictions not found.")

    predictions = np.fromfile(predictions_file, dtype=np.float16)
    predictions = predictions.reshape([-1, 20])

    print("A")
    m = truncated_dense_median(predictions)
    print(m)
    print(1/m)

    print("B")
    m = truncated_dense_median(predictions, random_choice=True)
    print(m)
    print(1/m)

    sample_medians = []

    for i in tqdm.tqdm(range(100)):
        print("C", i)
        m = sampled_median(predictions, n_samples=int(1e7))
        print(m)
        print(1/m)
        sample_medians.append(m)

    sample_medians = np.asarray(sample_medians)
    print("Mean:", np.mean(sample_medians))


if __name__ == "__main__":
    main()
