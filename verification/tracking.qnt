// One tick is one test millisecond. Hold lasts 2 ticks, return lasts 3.
module tracking {
  type Input = { signal: str, quality: str, calibrated: bool, dt: int }
  var event: Input
  var phase: str
  var elapsed: int
  var effect: str
  var valid: bool

  pure def frame(signal: str, quality: str, calibrated: bool, dt: int): Input =
    { signal: signal, quality: quality, calibrated: calibrated, dt: dt }
  action init = all {
    phase' = "Searching", elapsed' = 0, effect' = "none", valid' = true,
    event' = frame("None", "missing", false, 0),
  }
  action tick(input: Input): bool = {
    val present = input.quality != "missing"
    val reconnect = present and (input.quality == "usable" or input.signal == "Acquire")
    val acquiringFromLoss = Set("Searching", "LostHold", "ReturningNeutral").contains(phase)
    val lost = input.signal == "Degrade" and not(present)
    val age = elapsed + input.dt
    val dest =
      if (acquiringFromLoss and reconnect) "Acquiring"
      else if (phase == "LostHold" and age >= 2) "ReturningNeutral"
      else if (phase == "ReturningNeutral" and age >= 3) "Searching"
      else if (phase == "Acquiring" and present and input.calibrated and input.signal != "Degrade") "Tracking"
      else if (phase == "Degraded" and present and input.signal == "Acquire") "Tracking"
      else if (Set("Acquiring", "Tracking", "Degraded").contains(phase) and lost) "LostHold"
      else if (phase == "Tracking" and input.signal == "Degrade") "Degraded"
      else phase
    val actionName =
      if (dest == phase) "none"
      else if (dest == "LostHold") "StartHold"
      else if (dest == "ReturningNeutral") "StartReturnToNeutral"
      else if (dest == "Acquiring" or (phase == "Acquiring" and dest == "Tracking" and input.signal == "Acquire")) "ResetFilters"
      else "none"
    val time =
      if (dest == phase) age
      else if (dest == "ReturningNeutral") age - 2
      else if (dest == "Searching") age - 3
      else 0
    all {
      phase' = dest, elapsed' = if (time > 4) 4 else time, effect' = actionName, event' = input,
      valid' = and {
        (acquiringFromLoss and reconnect) implies (dest == "Acquiring" and actionName == "ResetFilters"),
        (phase == "Acquiring" and dest == "Tracking") implies (present and input.calibrated),
        (phase == "LostHold" and not(reconnect) and age >= 2) implies (dest == "ReturningNeutral" and time == age - 2),
        (phase == "ReturningNeutral" and not(reconnect) and age >= 3) implies (dest == "Searching" and time == age - 3),
        (dest == "LostHold" and dest != phase) implies (lost and actionName == "StartHold"),
        phase == dest implies actionName == "none",
      },
    }
  }
  action step = {
    nondet signal = Set("Acquire", "Degrade", "None").oneOf()
    nondet quality = Set("missing", "weak", "usable").oneOf()
    nondet calibrated = Bool.oneOf()
    nondet dt = Set(0, 1, 2, 4).oneOf()
    tick(frame(signal, quality, calibrated, dt))
  }
  val safety = valid and elapsed >= 0 and (phase == "LostHold" implies elapsed < 2)
  run dropoutTest = init.then(tick(frame("None", "usable", false, 1)))
    .then(tick(frame("None", "weak", true, 1))).then(tick(frame("Degrade", "missing", true, 1)))
    .then(tick(frame("None", "missing", true, 4))).expect(phase == "ReturningNeutral" and elapsed == 2)
    .then(tick(frame("None", "missing", true, 1))).expect(phase == "Searching")
  run dimReacquisitionTest = init.then(tick(frame("Acquire", "weak", false, 1)))
    .then(tick(frame("Acquire", "weak", true, 1))).then(tick(frame("Degrade", "weak", true, 1)))
    .then(tick(frame("Degrade", "missing", true, 1))).then(tick(frame("None", "usable", true, 4)))
    .expect(phase == "Acquiring" and elapsed == 0 and effect == "ResetFilters")
}
