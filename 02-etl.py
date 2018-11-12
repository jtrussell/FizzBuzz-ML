"""
Get our raw data into a format suitable for learning.

Each row will have 33 integer columns.

The first should be in the range 0-3 and indicates the how fizzy and buzzy that
number is:
    0 - No fizz, no buzz :(
    1 - Fizz!
    2 - Buzz!
    3 - FizzBuzz!

(An alternative implemenation could use two columns one for on/off of Fizz, the
other for on/off of Buzz. That's left as an exercise for the reader :wink:).

The remaining 32 columns make up a (right aligned) 32-bit representation of the
number being fizzbuzzed.

We'll output two files (identically formatted), one for training, the other for
testing.
"""

import numpy as np
import pandas as pd


def transform_row(row):
    """
    Takes in a row of raw output and spits out a processed row as described
    above.
    """
    cls = 0
    if row.Fizz == 'Fizz':
        cls = cls + 1
    if row.Buzz == 'Buzz':
        cls = cls + 2
    ret = [cls]
    num = bin(row.Num)[2:]
    num = '0' * (32 - len(num)) + num
    ret.extend(list(num))
    return ret


def main():
    raw_df = pd.read_csv('./data/raw.txt',
                         sep='\t',
                         dtype={'Num': np.int32, 'Fizz': str, 'Buzz': str})
    out_df = raw_df.apply(transform_row, axis=1, result_type='expand')

    # Use roughly 80% for training
    mask = np.random.rand(len(out_df)) < 0.8
    train_df = out_df[mask]
    test_df = out_df[~mask]

    train_df.to_csv('./data/training.txt',
                    header=False,
                    index=False,
                    sep='\t')

    test_df.to_csv('./data/testing.txt',
                   header=False,
                   index=False,
                   sep='\t')


if __name__ == '__main__':
    main()
