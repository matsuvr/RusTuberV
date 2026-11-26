module calibration_session {
  var event: str
  var phase: str
  var profile: str
  var accepted: bool
  action init = all { event' = "reset", phase' = "NotStarted", profile' = "", accepted' = true }
  action take(op: str): bool = {
    val ready = Set("A","B").contains(op) and phase == "Collecting"
    val dest = if (op == "reset") "NotStarted" else if (op == "start" and phase == "NotStarted") "Collecting"
      else if (ready) "Ready" else if (op == "reject" and phase == "Collecting") "Rejected"
      else if (op == "complete" and phase == "Ready") "Completed" else phase
    all {
      event' = op, phase' = dest, accepted' = (dest != phase or op == "reset"),
      profile' = if (ready) op else if (op == "reset") "" else profile,
    }
  }
  action step = { nondet op = Set("reset","start","reject","A","B","complete").oneOf() take(op) }
  val safety = (profile != "") == Set("Ready","Completed").contains(phase) and
    ((event == "complete" and accepted) implies phase == "Completed")
  run commitTest = init.then(take("start")).then(take("A")).then(take("B")).expect(not(accepted))
    .then(take("complete")).expect(phase == "Completed" and profile == "A")
    .then(take("reset")).then(take("start")).then(take("reject")).then(take("complete")).expect(not(accepted))
}
