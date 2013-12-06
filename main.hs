module Main( main ) where

import Perceptron
import System.Environment( getArgs )

main = do
  let xorData = [[0, 0, 0.05],  [0, 1, 0.95],  [1, 0, 0.95],  [1, 1, 0.05]] :: [[Double]]
  let (ins, outs) = unzip $ map (splitAt 2) xorData
  let (actFunc, actFunc') = (Perceptron.sigmoid, Perceptron.sigmoid')

  putStrLn "Fluffy Perceptron v0.1"

  args <- getArgs
  if (length args) /= 5
  then fail "5 parameters expected: LAYERS_SIZES RANDOM_INIT_RANGE ITER_LIMIT ERROR_LIMIT LEARNING_RATE"
  else do
    let sizes = read (args !! 0) :: [Int]
    let range = read (args !! 1) :: Double
    let iterLimit = read (args !! 2) :: Int
    let errorLimit = read (args !! 3) :: Double
    let learningRate = read (args !! 4) :: Double
    
    putStrLn "\nins = "
    print ins

    putStrLn "\ndesired outs = "
    print outs

    net <- Perceptron.initNet sizes range
    putStrLn "\nnet random initialized weights = "
    print net

    net <- Perceptron.trainNet iterLimit errorLimit learningRate actFunc actFunc' net $ zip ins outs
    putStrLn "\nnet trained weights = "
    print net

    putStrLn "\ntrained net predictions:"
    let prediction = Perceptron.predict actFunc net ins
    print prediction

    putStrLn "\ntrained net error:"
    print $ maximum $ maximum $ zipWith (\x y -> zipWith (\a b -> abs(a - b)) x y) prediction outs

