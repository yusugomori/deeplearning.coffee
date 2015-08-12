class @LogisticRegression
  constructor: (N, nIn, nOut, W, b) ->
    @N = N
    @nIn = nIn
    @nOut = nOut
    @W = new Array()
    @b = new Array()

    for i in [0...nOut] by 1
      @W[i] = new Array()
      @b[i] = 0.0

      for j in [0...nIn] by 1
        @W[i][j] = 0.0


  train: (x, y, lr, dy) ->
    pYGivenX = (0.0 for [0...@nOut])

    if dy is null
      dy = (0.0  for [0...@nOut])

    for i in [0...@nOut] by 1
      pYGivenX[i] = 0.0

      for j in [0...@nIn] by 1
        pYGivenX[i] += @W[i][j] * x[j]

      pYGivenX[i] += @b[i]

    @softmax pYGivenX

    for i in [0...@nOut] by 1

      dy[i] = y[i] - pYGivenX[i]

      for j in [0...@nIn] by 1
        @W[i][j] += lr * dy[i] * x[j] / @N

      @b[i] += lr * dy[i] / @N

    return


  softmax: (x) ->
    max = 0.0
    sum = 0.0

    for i in [0...@nOut] by 1
      if max < x[i]
        max = x[i]

    for i in [0...@nOut] by 1
      x[i] = Math.exp( x[i] - max )
      sum += x[i]

    for i in [0...@nOut] by 1
      x[i] /= sum

    return


  predict: (x, y) ->
    for i in [0...@nOut] by 1
      y[i] = 0.0

      for j in [0...@nIn] by 1
        y[i] += @W[i][j] * x[j]

      y[i] += @b[i]

    @softmax y



@test_LogisticRegression = () ->
  learningRate = 0.1
  nEpochs = 500

  trainN = 4
  testN = 4
  nIn = 2
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
    [0, 1],
    [1, 0],
    [1, 0]
  ]


  # construct LogisticRegresson
  classifier = new LogisticRegression(trainN, nIn, nOut)

  # train
  for epoch in [0...nEpochs] by 1
    for i in [0...trainN] by 1
      classifier.train trainX[i], trainY[i], learningRate, null

  # test data
  testX = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ]

  testY = new Array()

  for i in [0...testN] by 1
    testY[i] = new Array()
    for j in [0...nOut] by 1
      testY[i][j] = 0.0


  # test
  for i in [0...testN] by 1
    log = ''

    classifier.predict testX[i], testY[i]

    for j in [0...nOut] by 1
      log +=  testY[i][j] + " "
    console.log log
