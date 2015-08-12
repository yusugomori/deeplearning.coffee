class @Dropout
  constructor: (N, nIn, hiddenLayerSizes, nOut, activaton) ->
    @N = N
    @nIn = nIn
    @hiddenLayerSizes = hiddenLayerSizes
    @nLayers = hiddenLayerSizes.length
    @nOut = nOut

    @hiddenLayers = new Array()

    activaton = ReLU if activaton is null

    # construct multi-layer
    for i in [0...@nLayers] by 1
      if i == 0
        inputSize = nIn
      else
        inputSize = hiddenLayerSizes[i-1]

      # construct hiddenLayer
      @hiddenLayers.push new HiddenLayer(N, inputSize, hiddenLayerSizes[i], null, null, activaton)

    # construct logisticLayer
    @logisticLayer = new LogisticRegression(N, hiddenLayerSizes[@nLayers-1], nOut)


  train: (epochs, trainX, trainY, dropout, pDropout, lr) ->

    for epoch in [0...epochs] by 1
      for n in [0...@N] by 1

        dropoutMasks = new Array()
        layerInputs = new Array()

        # forward hiddenLayers
        for i in [0...@nLayers] by 1

          if i == 0
            layerInput = trainX[n]
          else
            layerInput = layerOutput.concat()

          layerInputs.push layerInput.concat()

          layerOutput = (0.0 for [0...@hiddenLayerSizes[i]])
          @hiddenLayers[i].forward layerInput, layerOutput

          if dropout is true
            mask = @hiddenLayers[i].dropout layerOutput.length, pDropout
            for j in [0...layerOutput.length] by 1
              layerOutput[j] *= mask[j]

            dropoutMasks.push mask.concat()

        # forward & backward logisticLayer
        logisticLayerDy = (0.0 for [0...@nOut])
        @logisticLayer.train layerOutput, trainY[n], lr, logisticLayerDy
        layerInputs.push layerOutput.concat()


        # backward hiddenLayers
        for i in [@nLayers-1..0] by -1

          if i == @nLayers-1
            prevDy = logisticLayerDy
            prevW = @logisticLayer.W
          else
            prevDy = dy.concat()
            prevW = @hiddenLayers[i+1].W

          dy = (0.0 for [0...@hiddenLayerSizes[i]])
          @hiddenLayers[i].backward layerInputs[i], dy, layerInputs[i+1], prevDy, prevW, lr

          if dropout is true
            for j in [0...dy.length] by 1
              dy[j] *= dropoutMasks[i][j]

    return


  pretest: (pDropout) ->
    for i in [0...@nLayers] by 1
      if i == 0
        inn = @nIn
      else
        inn = @hiddenLayerSizes[i]

      if i == @nLayers-1
        out = @nOut
      else
        out = @hiddenLayerSizes[i+1]

      for l in [0...out] by 1
        for m in [0...inn] by 1
          @hiddenLayers[i].W[l][m] *= 1 - pDropout

    return


  predict: (x, y) ->
    for i in [0...@nLayers] by 1
      if i == 0
        layerInput = x
      else
        layerInput = layerOutput.concat()

      layerOutput = (0.0 for [0...@hiddenLayerSizes[i]])
      @hiddenLayers[i].forward layerInput, layerOutput

    @logisticLayer.predict layerOutput, y

    return


@test_Dropout = () ->
  @RAND_SEED = 1234

  learningRate = 0.1
  nEpochs = 3000

  trainN = 4
  testN = 4
  nIn = 2
  hiddenLayerSizes = [15, 12]
  nOut = 2

  dropout = true
  pDropout = 0.5


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

  # construct Dropout
  classifier = new Dropout(trainN, nIn, hiddenLayerSizes, nOut, ReLU)

  # train
  classifier.train nEpochs, trainX, trainY, dropout, pDropout, learningRate


  # pretest
  if dropout is true
    classifier.pretest pDropout

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
