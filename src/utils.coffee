
@rand = () ->
  if @RAND_SEED is null
    @RAND_SEED = 1234

  x = Math.sin(@RAND_SEED++) * 1000000;

  return x - Math.floor x


@uniform = (min, max) ->
  return @rand() * (max - min) + min

@binomial = (n, p) ->
  if p < 0 or p > 1
    return 0

  c = 0
  for i in [0...n] by 1
    r = @rand()
    c += 1 if r < p

  return c


@sigmoid = (x) ->
  return 1 / (1 + Math.pow(Math.E, -x))

@dsigmoid = (y) ->
  return y * (1 - y)


@tanh = (x) ->
  return Math.tanh(x)

@dtanh = (y) ->
  return 1 - y * y


@ReLU = (x) ->
  if x > 0
    return x
  else
    return 0

@dReLU = (y) ->
  if y > 0
    return 1
  else
    return 0
