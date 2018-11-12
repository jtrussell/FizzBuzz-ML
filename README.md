
# FizzBuzz

> Solving FizzBuzz with neural networks. That's deep.

For the next time you get asked to whiteboard this problem.


## Getting Started

```
conda env create -f environment.yml
activate fizzbuzz
```

Note that second line may need to be `source activate fizzbuzz` depending on
your platform.


## Training Pipeline

```shell
# Generate some raw data
python 01-make-raw-data.py

# Prep our data model input
python 02-etl.py

# Train
python 03-train.py

# Validate
python 04-validate.py
```

## License

MIT

This project is a toy, have fun playing with it :).
