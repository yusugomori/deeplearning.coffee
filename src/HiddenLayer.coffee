class @HiddenLayer
  constructor: (N, nIn, nOut, W, b, activation) ->
    @N = N
    @nIn = nIn
    @nOut = nOut

    if W is null
      @W = new Array()
      a = 1.0 / nIn

      for i in [0...nOut] by 1
        @W[i] = new Array()

        for j in [0...nIn] by 1
          @W[i][j] = uniform(-a, a)

    else
      @W = W

    if b is null
      @b = (0.0 for [0...nOut])
    else
      @b = b

    if activation is sigmoid
      @dactivation = dsigmoid

    else if activation is tanh
      @dactivation = dtanh

    else if activation is ReLU
      @dactivation = dReLU

    else
      throw new Error 'activation function not supported'

    @activation = activation


  output: (input, w, b) ->
    linearOutput = 0.0

    for j in [0...@nIn] by 1
      linearOutput += w[j] * input[j]

    linearOutput += b

    return @activation linearOutput


  forward: (input, output) ->
    for i in [0...@nOut] by 1
      output[i] = @output( input, @W[i], @b[i] )

    return


  backward: (input, dy, prevLayerInput, prevLayerDy, prevLayerW, lr) ->

    if dy is null
      dy = (0.0 for [0...@nOut])

    prevNIn = @nOut
    prevNOut = prevLayerDy.length

    for i in [0...prevNIn] by 1
      dy[i] = 0

      for j in [0...prevNOut] by 1
        dy[i] += prevLayerDy[j] * prevLayerW[j][i]

      dy[i] *= @dactivation prevLayerInput[i]

    for i in [0...@nOut] by 1
      for j in [0...@nIn] by 1
        @W[i][j] += lr * dy[i] * input[j] / @N

      @b[i] += lr * dy[i] / @N

    return


  dropout: (size, p) ->
    mask = []

    for i in [0...size] by 1
      mask.push binomial 1, p

    return mask