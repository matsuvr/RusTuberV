// A coordinate of an already certified joint segment. Exact fixed-point
// arithmetic keeps both endpoint directions and the eased curve comparable.
module arm_path {
  pure val ticks = 8
  pure val scale = ticks^5
  var startAngle: int
  var endAngle: int
  var eased: bool
  var tick: int
  var delta: int
  var angle: int
  var monotone: bool
  action init = {
    nondet a = (-2).to(2).oneOf()
    nondet b = (-2).to(2).oneOf()
    nondet ease = Bool.oneOf()
    all { startAngle' = a, endAngle' = b, eased' = ease, tick' = 0, delta' = 0, angle' = a * scale, monotone' = true }
  }
  action advance(dt: int): bool = {
    val t = if (tick + dt >= ticks) ticks else tick + dt
    val weight = if (eased) t^3 * (10 * ticks^2 - 15 * ticks * t + 6 * t^2) else t * ticks^4
    val sample = startAngle * scale + (endAngle - startAngle) * weight
    all {
      startAngle' = startAngle, endAngle' = endAngle, eased' = eased, tick' = t, delta' = dt, angle' = sample,
      monotone' = if (startAngle <= endAngle) sample >= angle else sample <= angle,
    }
  }
  action step = { nondet dt = 0.to(ticks + 1).oneOf() advance(dt) }
  val safety = and {
    monotone,
    if (startAngle < endAngle) angle >= startAngle * scale and angle <= endAngle * scale
    else angle <= startAngle * scale and angle >= endAngle * scale,
    tick == ticks implies angle == endAngle * scale,
  }
  run endpointTest = init.then(advance(3)).then(advance(6)).expect(tick == ticks and angle == endAngle * scale)
}
