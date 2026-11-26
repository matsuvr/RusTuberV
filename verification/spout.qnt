// An upload can return after stop. The worker starts at its native send boundary.
module spout {
  var event: str
  var phase: str
  var pending: int
  var sent: int
  var overwritten: bool
  action init = all { event' = "init", phase' = "Starting", pending' = 0, sent' = 0, overwritten' = false }
  action take(op: str): bool = {
    val id = if (op == "A") 1 else if (op == "B") 2 else 0
    val drain = op == "send" or op == "stopInFlight"
    val close = op == "stop" or op == "stopInFlight"
    all {
      drain implies pending != 0,
      phase' = if (close) "Off" else if (drain) "Live" else phase,
      sent' = if (drain) pending else sent,
      pending' = if (close or drain or op == "discard") 0 else if (phase != "Off") id else 0,
      overwritten' = (id != 0 and pending != 0), event' = op,
    }
  }
  action step = { nondet op = Set("A","B","discard","stop","send","stopInFlight").oneOf() take(op) }
  val safety = and {
    Set("stop","stopInFlight").contains(event) implies phase == "Off",
    phase == "Off" implies pending == 0, phase == "Live" implies sent != 0,
    overwritten implies (Set("A","B").contains(event) and pending != 0),
  }
  run stopInFlightTest = init.then(take("A")).then(take("B")).expect(overwritten)
    .then(take("send")).expect(sent == 2 and phase == "Live")
    .then(take("B")).then(take("discard")).then(take("A")).then(take("stopInFlight")).then(take("B"))
    .expect(phase == "Off" and sent == 1 and pending == 0)
}
