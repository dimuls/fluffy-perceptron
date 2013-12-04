import System.Random
import Data.Random
import Data.List
import Control.Applicative
import Control.Monad



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

initWeights :: Int -> Int -> Double -> IO [[Double]]
initWeights inSize outSize range = replicateM outSize $ replicateM inSize $ randomRIO (-range, range)

nnInit :: [Int] -> Double -> IO [[[Double]]]
nnInit sizes range = mapM (\[inSize, outSize] -> initWeights (inSize+1) outSize range ) $ pairs sizes



-- Forward propogation

passLayer :: (Double -> Double) -> [Double] -> [[Double]] -> [Double]
passLayer actFunc input weigths =
  map actFunc $ map sum $ map (zipWith (*) input) weigths

passLayers :: (Double -> Double) -> [[[Double]]] -> [Double] -> [[Double]]
passLayers actFunc [] input = []
passLayers actFunc (inputWeigths:layers) input = 
  layerOut:(passLayers actFunc layers layerOut)
  where
    layerOut = passLayer actFunc (1:input) inputWeigths

nnForwardPropagate :: (Double -> Double) -> [[[Double]]] -> [Double] -> [[Double]]
nnForwardPropagate actFunc nn input =
  input:passLayers actFunc nn input

predict' :: (Double -> Double) -> [[[Double]]] -> [Double] -> [Double]
predict' actFunc nn input = 
  last $ passLayers actFunc nn input

predict :: (Double -> Double) -> [[[Double]]] -> [[Double]] -> [[Double]]
predict actFunc nn inputs = 
  map (predict' actFunc nn) inputs



-- Back propagation

layerCorrection :: Double -> [Double] -> [Double] -> [[Double]]
layerCorrection learningRate previousGradient layerInput = 
  map (\localGradient -> map (* (learningRate * localGradient)) layerInput) previousGradient

backpassLayers :: Double -> (Double -> Double) -> [[[Double]]] -> [[Double]] -> [Double] -> [[[Double]]]
backpassLayers learningRate actFunc' [] [] previousGradient = []
backpassLayers learningRate actFunc' (weights:nn') (layerInput:outputs') previousGradient =
  correction:backpassLayers learningRate actFunc' nn' outputs' gradient
  where
    correction = layerCorrection learningRate previousGradient (1:layerInput)
    gradient = zipWith (\neuronInput' weights -> neuronInput' * (sum (zipWith (*) previousGradient weights))) layerInput' weights'
    weights' = tail $ transpose weights
    layerInput' = map actFunc' layerInput

nnBackPropagate :: Double -> (Double -> Double) -> [[[Double]]] -> [[Double]] -> [Double] -> [[[Double]]]
nnBackPropagate learningRate actFunc' nn outputs desiredOutput =
  reverse $ backpassLayers learningRate actFunc' nn' outputs' gradient
  where
    gradient = zipWith (\out dout -> (dout - out) * (actFunc' out)) lastOutput desiredOutput
    (lastOutput:outputs') = reverse outputs
    nn' = reverse nn



-- Train neural network

nnSingleTrain :: Double -> (Double -> Double) -> (Double -> Double) -> [[[Double]]] -> [[Double]] -> [[Double]] -> [[[Double]]]
nnSingleTrain _ _ _ nn [] [] = nn 
nnSingleTrain learningRate actFunc actFunc' nn (input:inputs) (desiredOutput:desiredOutputs) =
  nnSingleTrain learningRate actFunc actFunc' nn' inputs desiredOutputs
  where
    nn' = (zipWith $ zipWith $ zipWith (+)) nn correction
    correction = nnBackPropagate learningRate actFunc' nn outputs desiredOutput
    outputs = nnForwardPropagate actFunc nn input

train :: Integer -> Double -> Double -> (Double -> Double) -> (Double -> Double) -> [[[Double]]] -> [([Double], [Double])] -> IO [[[Double]]]
train 0 _ _ _ _ nn _ = do return nn
train iterationLimit errorLimit learningRate actFunc actFunc' nn trainData = do
  shuffledTrainData <- runRVar (shuffle trainData) StdRandom
  let (inputs, outputs) = unzip shuffledTrainData
  let nn' = nnSingleTrain learningRate actFunc actFunc' nn inputs outputs
  let error = maximum $ map maximum $ zipWith (\x y -> zipWith (\a b -> abs(a - b)) x y) (predict actFunc nn' inputs) outputs
  if error <= errorLimit
    then return nn'
    else train (iterationLimit-1) errorLimit learningRate actFunc actFunc' nn' trainData





main = do
  let xorData = [[0, 0, 0.05],  [0, 1, 0.95],  [1, 0, 0.95],  [1, 1, 0.05]] :: [[Double]]
  let (ins, outs) = unzip $ map (splitAt 2) xorData
  let (actFunc, actFunc') = (sigmoid, sigmoid')

  putStrLn "Fluffy Perceptron v0.1"

  putStrLn "\nins = "
  print ins

  putStrLn "\ndesired outs = "
  print outs

  nn <- nnInit [2, 2, 1] 1
  putStrLn "\nnn random initialized weights = "
  print nn

  nn <- train 1000 0.01 1 actFunc actFunc' nn $ zip ins outs
  putStrLn "\nnn trained weights = "
  print nn

  putStrLn "\ntrained nn predictions:"
  let pred = predict actFunc nn ins
  print pred

  putStrLn "\ntrained nn error:"
  print $ maximum $ zipWith (\x y -> zipWith (\a b -> abs(a - b)) x y) (predict actFunc nn ins) outs
