Fluffy Perceptron v0.1
=================

Simple working perceptron on haskell

There is one main.hs file available to run. In main function you can see perceptron initialization, training and using for XOR problem solving.

You can run main.hs in the following way:
```
$ runhaskell main.hs LAYERS_SIZES RANDOM_INIT_RANGE ITER_LIMIT ERROR_LIMIT LEARNING_RATE
```
where:
- LAYER_SIZES: list of integers, representing number of nodes in each layer, for example [1,5,3];
- RANDOM_INIT_RANGE: double, for exmaple 1;
- ITER_LIMIT: integer, maximum number of iterations;
- ERROR_LIMIT: double, needed max error;
- LEARNING_RATE: double.

So, for example you can run it in the following way:
```
$ runhaskell main.hs [2,4,1] 1 1000 0.01 1

```

# Dependencies
You need to install random-extras package:
```
$ cabal install random-extras
```
