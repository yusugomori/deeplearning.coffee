class @MLP
  constructor: (N, nIn, nHidden, nOut) ->
    @N = N
    @nIn = nIn
    @nOut = nOut

    @hiddenLayer = new HiddenLayer( N, nIn, nHidden, null, null, tanh)
    @logisticLayer = new LogisticRegression(N, nHidden, nOut)

  train: (trainX, trainY, lr) ->
    hiddenLayerInput = (0.0 for [0...@nIn])
    logisticLayerInput = (0.0 for [0...@nHidden])

    for n in [0...@N]

      for j in [0...@nIn]
        hiddenLayerInput[j] = trainX[n][j]

      # forward HiddenLayer
      @hiddenLayer.forward hiddenLayerInput, logisticLayerInput


      # forward and backward logisticLayer
      dy = (0.0 for [0...@nOut])
      @logisticLayer.train logisticLayerInput, trainY[n], lr, dy


      # backward hiddenLayer
      @hiddenLayer.backward hiddenLayerInput, null, logisticLayerInput, dy, @logisticLayer.W, lr

    return

  predict: (x, y) ->
    logisticLayerInput = (0.0 for [0...@nHidden])

    @hiddenLayer.forward x, logisticLayerInput
    @logisticLayer.predict logisticLayerInput, y

    return


@test_MLP = () ->
  @RAND_SEED = 1234

  learningRate = 0.1
  nEpochs = 2000

  trainN = 4
  testN = 4
  nIn = 2
  nHidden = 3
  nOut = 2


  # train XOR
  trainX = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ]

  trainY = [
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
  ]

  # construct MLP
  classifier = new MLP(trainN, nIn, nHidden, nOut)

  # train
  for epoch in [0...nEpochs]
    classifier.train trainX, trainY, learningRate

  # test data
  testX = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ]

  testY = new Array()

  for i in [0...testN]
    testY[i] = new Array()
    for j in [0...nOut]
      testY[i][j] = 0.0


  # test
  for i in [0...testN]
    log = ''

    classifier.predict testX[i], testY[i]

    for j in [0...nOut]
      log +=  testY[i][j] + " "
    console.log log
