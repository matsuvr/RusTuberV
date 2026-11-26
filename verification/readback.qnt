// One outstanding GPU callback. For its two generation tokens, equality to
// the current token is sufficient; a restart cannot make an old token fresh.
module readback {
  var active: bool
  var preview: bool
  var sleeping: bool
  var request: bool
  var sameEpoch: bool
  var samePower: bool
  var delivered: bool
  var event: str
  action init = all {
    active' = false, preview' = false, sleeping' = false, request' = false,
    sameEpoch' = false, samePower' = false, delivered' = false, event' = "init",
  }
  action take(op: str): bool = {
    val outputOn = if (op == "on") true else if (op == "off") false else active
    val asleep = if (op == "sleep") true else if (op == "wake") false else sleeping
    val issue = op == "issue" and active and not(sleeping)
    all {
      active' = outputOn, sleeping' = asleep, event' = op,
      preview' = if (op == "show") true else if (op == "hide") false else preview,
      request' = (request or issue),
      sameEpoch' = (issue or (sameEpoch and outputOn == active)),
      samePower' = (issue or (samePower and not(asleep and not(sleeping)))),
      delivered' = (op == "complete" and request and active and not(sleeping) and sameEpoch and samePower),
    }
  }
  action step = {
    nondet op = Set("on", "off", "show", "hide", "sleep", "wake", "issue", "complete").oneOf()
    take(op)
  }
  val safety = delivered implies (active and not(sleeping) and request and sameEpoch and samePower)
  run lateCallbackTest = init.then(take("show")).then(take("on")).then(take("issue"))
    .then(take("sleep")).then(take("wake")).then(take("complete")).expect(not(delivered))
    .then(take("issue")).then(take("complete")).expect(delivered)
    .then(take("off")).then(take("on")).then(take("complete")).expect(not(delivered))
}
