"""
Do some raw fizzin and buzzin
"""


def fizzbuzz(num):
    """
    A pretty vanilla actual fizzbuzz solution to build our raw dataset.
    """
    if num % 3 == 0:
        if num % 5 == 0:
            return ('Fizz', 'Buzz')
        else:
            return('Fizz', '')
    if num % 5 == 0:
        return ('', 'Buzz')
    return ('', '')


def main():
    with open('./data/raw.txt', 'w') as f:
        f.write('Num\tFizz\tBuzz')
        for n in range(1, int(1e5)):
            fizz, buzz = fizzbuzz(n)
            f.write('\n{}\t{}\t{}'.format(n, fizz, buzz))


if __name__ == '__main__':
    main()
