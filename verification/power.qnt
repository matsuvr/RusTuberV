// Up to three actual suspend cycles, starting just before the packed u64 wraps.
module power {
  var event: str
  var cycles: int
  var sleeping: bool
  var observed: int
  var observedSleep: bool
  action init = all { event' = "init", cycles' = 0, sleeping' = false, observed' = 0, observedSleep' = false }
  action take(op: str): bool = {
    val newCycle = op == "suspend" and not(sleeping)
    all {
      newCycle implies cycles < 3,
      event' = op, cycles' = if (newCycle) cycles + 1 else cycles,
      sleeping' = if (op == "observe") sleeping else op == "suspend",
      observed' = if (op == "observe") cycles else observed,
      observedSleep' = if (op == "observe") sleeping else observedSleep,
    }
  }
  action step = { nondet op = Set("suspend","resume","observe").oneOf() take(op) }
  val safety = observed <= cycles and cycles <= 3 and
    (event == "observe" implies (observed == cycles and observedSleep == sleeping))
  run missedWakeTest = init.then(take("suspend")).then(take("suspend")).then(take("resume"))
    .then(take("resume")).then(take("suspend")).then(take("resume")).then(take("observe"))
    .expect(cycles == 2 and observed == 2 and not(observedSleep))
}
