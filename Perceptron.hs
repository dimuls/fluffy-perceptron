module Perceptron (
  sigmoid,
  sigmoid',
  initNet,
  trainNet,
  predict
) where

import System.Random
import Data.Random
import Data.List
import Control.Applicative
import Control.Monad

type Weight = Double
type Neuron = [Weight]
type Layer = [Neuron]
type Net = [Layer]



-- Activation functions

threshold :: Double -> Double
threshold x
  | x > 0 = 1
  | otherwise = 0

threshold' :: Double -> Double
threshold' x = x

sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

sigmoid' :: Double -> Double
sigmoid' x = (1 - x) * x



-- Neural network initialize

pairs :: [a] -> [[a]]
pairs (x : y : xs) = [x,y] : pairs (y:xs)
pairs _ = []

initLayer :: Int -> Int -> Double -> IO Layer
initLayer inSize outSize range = replicateM outSize $ replicateM inSize $ randomRIO (-range, range)

initNet :: [Int] -> Double -> IO Net
initNet sizes range = mapM (\[inSize, outSize] -> initLayer (inSize+1) outSize range ) $ pairs sizes



-- Forward propogation

passLayer :: (Double -> Double) -> [Double] -> Layer -> [Double]
passLayer actFunc input weigths =
  map (actFunc . sum) $ map (zipWith (*) input) weigths

passNet :: (Double -> Double) -> Net -> [Double] -> [[Double]]
passNet actFunc [] input = []
passNet actFunc (inputLayer:layers) input = 
  layerOut:(passNet actFunc layers layerOut)
  where
    layerOut = passLayer actFunc (1:input) inputLayer

forwardPropagate :: (Double -> Double) -> Net -> [Double] -> [[Double]]
forwardPropagate actFunc net input =
  input:passNet actFunc net input

predict' :: (Double -> Double) -> Net -> [Double] -> [Double]
predict' actFunc net input = 
  last $ passNet actFunc net input

predict :: (Double -> Double) -> Net -> [[Double]] -> [[Double]]
predict actFunc net inputs = 
  map (predict' actFunc net) inputs



-- Back propagation

layerCorrection :: Double -> [Double] -> [Double] -> [[Double]]
layerCorrection learningRate previousGradient layerInput = 
  map (\localGradient -> map (* (learningRate * localGradient)) layerInput) previousGradient

backpassNet :: Double -> (Double -> Double) -> Net -> [[Double]] -> [Double] -> [[[Double]]]
backpassNet learningRate actFunc' [] [] previousGradient = []
backpassNet learningRate actFunc' (layer:net') (input:inputs) previousGradient =
  correction:backpassNet learningRate actFunc' net' inputs gradient
  where
    correction = layerCorrection learningRate previousGradient (1:input)
    gradient = zipWith (\input weights -> (actFunc' input) * (sum (zipWith (*) previousGradient weights))) input layer'
    layer' = tail $ transpose layer

backwardPropagate :: Double -> (Double -> Double) -> Net -> [[Double]] -> [Double] -> [[[Double]]]
backwardPropagate learningRate actFunc' net outputs desiredOutput =
  reverse $ backpassNet learningRate actFunc' net' layersInputs gradient
  where
    gradient = zipWith (\out dout -> (dout - out) * (actFunc' out)) netOutput desiredOutput
    (netOutput:layersInputs) = reverse outputs
    net' = reverse net



-- Train neural network

singleTrain :: Double -> (Double -> Double) -> (Double -> Double) -> Net -> [[Double]] -> [[Double]] -> [[[Double]]]
singleTrain _ _ _ net [] [] = net 
singleTrain learningRate actFunc actFunc' net (input:inputs) (desiredOutput:desiredOutputs) =
  singleTrain learningRate actFunc actFunc' net' inputs desiredOutputs
  where
    net' = (zipWith $ zipWith $ zipWith (+)) net correction
    correction = backwardPropagate learningRate actFunc' net outputs desiredOutput
    outputs = forwardPropagate actFunc net input

trainNet :: Int -> Double -> Double -> (Double -> Double) -> (Double -> Double) -> Net -> [([Double], [Double])] -> IO Net
trainNet 0 _ _ _ _ net _ = do return net
trainNet iterLimit errorLimit learningRate actFunc actFunc' net trainData = do
  shuffledTrainData <- runRVar (shuffle trainData) StdRandom
  let (inputs, outputs) = unzip shuffledTrainData
  let net' = singleTrain learningRate actFunc actFunc' net inputs outputs
  let error = maximum $ maximum $ zipWith (\x y -> zipWith (\a b -> abs(a - b)) x y) (predict actFunc net' inputs) outputs
  if error <= errorLimit
    then return net'
    else trainNet (iterLimit-1) errorLimit learningRate actFunc actFunc' net' trainData
