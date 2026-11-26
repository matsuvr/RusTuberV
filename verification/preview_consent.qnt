module preview_consent {
  var event: str
  var pane: str
  var controls: bool
  var phase: str
  action init = all { event' = "init", pane' = "camera", controls' = true, phase' = "Hidden" }
  action take(op: str): bool = {
    val draw = Set("camera","other").contains(op)
    val revoke = Set("hide","close","open","select","unload","navigate","import").contains(op) or (draw and pane != op)
    all {
      Set("request","confirm").contains(op) implies (controls and pane == "camera"),
      event' = op, controls' = if (op == "open") true else if (op == "close") false else controls,
      pane' = if (draw) op else pane,
      phase' = if (revoke) "Hidden" else if (op == "request" and phase == "Hidden") "Confirming"
        else if (op == "confirm" and phase == "Confirming") "Visible" else phase,
    }
  }
  action step = {
    nondet op = Set("camera","other","request","confirm","hide","close","open","select","unload","navigate","import").oneOf()
    take(op)
  }
  val safety = (phase != "Hidden" implies (controls and pane == "camera")) and
    (Set("navigate","unload","select","import","open","close").contains(event) implies phase == "Hidden")
  run revokeTest = init.then(take("confirm")).expect(phase == "Hidden").then(take("request")).then(take("camera"))
    .then(take("confirm")).expect(phase == "Visible").then(take("other")).then(take("camera")).then(take("confirm"))
    .expect(phase == "Hidden").then(take("request")).then(take("select")).then(take("confirm")).expect(phase == "Hidden")
}
