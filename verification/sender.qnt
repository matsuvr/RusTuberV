// The producer may submit several frames while NDI is starting. Once Live,
// each step ends after the scripted sender consumes its pending frame.
module sender {
  var phase: str
  var pending: int
  var lastSent: int
  var reply: str
  var event: str
  var valid: bool
  action init = all {
    phase' = "Off", pending' = 0, lastSent' = 0, reply' = "none", event' = "init", valid' = true,
  }
  action take(op: str): bool = {
    val id = if (op == "submitA") 1 else if (op == "submitB") 2 else 0
    val start = op == "start" and phase == "Off"
    val live = op == "ready" and phase == "Starting"
    val result =
      if (op == "start") { if (start) "started" else "busy" }
      else if (id == 0) "none"
      else if (phase == "Off") "RejectedNotRunning"
      else if (pending != 0) "Replaced"
      else "Submitted"
    val sent =
      if (start) 0
      else if (live and pending != 0) pending
      else if (phase == "Live" and id != 0) id
      else lastSent
    all {
      phase' = if (op == "stop") "Off" else if (start) "Starting" else if (live) "Live" else phase,
      pending' = if (op == "stop" or live) 0 else if (phase == "Starting" and id != 0) id else pending,
      lastSent' = sent, reply' = result, event' = op,
      valid' = and {
        op == "stop" implies sent == lastSent,
        (phase != "Live" and id != 0) implies sent == lastSent,
        (live and pending != 0) implies sent == pending,
        start implies sent == 0,
      },
    }
  }
  action step = { nondet op = Set("start", "ready", "stop", "submitA", "submitB").oneOf() take(op) }
  val safety = valid and (phase != "Starting" implies pending == 0)
    and (reply == "RejectedNotRunning" implies phase == "Off")
  run replacementTest = init.then(take("start")).then(take("submitA")).then(take("submitB"))
    .expect(reply == "Replaced").then(take("ready")).expect(lastSent == 2)
    .then(take("stop")).then(take("start")).then(take("submitA")).then(take("stop"))
    .then(take("start")).then(take("ready")).expect(lastSent == 0)
}
