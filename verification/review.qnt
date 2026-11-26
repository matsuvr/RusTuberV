// Interactive VRM import. Callback IDs represent the current request or any
// superseded one, including an earlier review of the very same path.
module review {
  var phase: str
  var file: int
  var checked: bool
  var event: str
  var authorized: bool
  action init = all {
    phase' = "Idle", file' = 0, checked' = false, event' = "init", authorized' = true,
  }
  action take(op: str): bool = {
    val choose = op == "chooseA" or op == "chooseB"
    val confirm = op == "confirm" and phase == "Review" and checked
    val cancel = op == "cancel" or op == "unload"
    val after =
      if (choose) "Reading"
      else if (cancel) "Idle"
      else if (op == "complete" and phase == "Reading") "Review"
      else if (op == "error" and phase == "Reading") "Failed"
      else if (confirm) "Importing"
      else phase
    all {
      phase' = after, event' = op,
      file' = if (choose) { if (op == "chooseA") 1 else 2 } else if (cancel) 0 else file,
      checked' = (after == "Review" and (op == "check" or (checked and op != "uncheck"))),
      authorized' = and {
        (after == "Importing" and phase != after) implies (phase == "Review" and checked and op == "confirm"),
        op == "stale" implies after == phase,
        cancel implies after == "Idle",
      },
    }
  }
  action step = {
    nondet op = Set("chooseA", "chooseB", "cancel", "unload", "complete", "error",
                    "stale", "check", "uncheck", "confirm").oneOf()
    take(op)
  }
  val safety = authorized and (checked implies phase == "Review") and ((file == 0) == (phase == "Idle"))
  run replacedReviewTest = init.then(take("chooseA")).then(take("complete")).then(take("check"))
    .then(take("chooseB")).then(take("stale")).then(take("confirm")).expect(phase == "Reading")
    .then(take("complete")).then(take("confirm")).expect(phase == "Review")
    .then(take("check")).then(take("confirm")).expect(phase == "Importing" and file == 2)
  run cancelledReviewTest = init.then(take("chooseA")).then(take("cancel")).then(take("stale"))
    .expect(phase == "Idle").then(take("chooseA")).then(take("stale")).expect(phase == "Reading")
}
