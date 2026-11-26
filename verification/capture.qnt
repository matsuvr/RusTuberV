// Capture worker after draining one command or handling one backend result.
// Stop keeps the selection, reset forgets it. Deadline passage is an input.
module capture {
  var phase: str
  var requested: bool
  var sleeping: bool
  var attempts: int
  var event: str
  var valid: bool
  action init = all {
    phase' = "Idle", requested' = false, sleeping' = false, attempts' = 0, event' = "init", valid' = true,
  }
  action take(op: str): bool = {
    val start = op == "startOk" or op == "startTransient" or op == "startFatal"
    val suspend = op == "sleep" and not(sleeping)
    val wake = op == "wake" and sleeping
    val retry = phase == "Reconnecting" and Set("wait", "retryOk", "retryErr").contains(op)
    val exhausted = retry and attempts >= 5
    val after =
      if (op == "reset") "Idle"
      else if (op == "stop") { if (phase == "Idle") "Idle" else "Selected" }
      else if (start and sleeping) "Suspended"
      else if (op == "startFatal" or exhausted) "BackOff"
      else if (op == "startTransient") "Reconnecting"
      else if (op == "startOk") "Running"
      else if (requested and suspend) "Suspended"
      else if (requested and (wake or (op == "missedCycle" and not(sleeping)))) "Reconnecting"
      else if (op == "disconnect" and phase == "Running") "Reconnecting"
      else if (retry and op == "retryOk") "Running"
      else phase
    val resetBudget = start or (requested and (suspend or wake or (op == "missedCycle" and not(sleeping))))
      or (phase == "Running" and op == "disconnect")
    all {
      phase' = after, event' = op,
      sleeping' = if (op == "sleep") true else if (op == "wake") false else sleeping,
      requested' = Set("Running", "Suspended", "Reconnecting").contains(after),
      attempts' = if (resetBudget) 0 else if (retry and not(exhausted) and op != "wait") attempts + 1 else attempts,
      valid' = and {
        ((wake or op == "missedCycle") and not(requested)) implies after == phase,
        (op == "stop" or op == "reset") implies Set("Idle", "Selected").contains(after),
        exhausted implies after == "BackOff",
        (retry and not(exhausted) and op == "wait") implies after == phase,
      },
    }
  }
  action step = {
    nondet op = Set("startOk", "startTransient", "startFatal", "reset", "stop", "sleep", "wake",
                    "missedCycle", "disconnect", "wait", "retryOk", "retryErr").oneOf()
    take(op)
  }
  val safety = valid and attempts <= 5 and (sleeping implies not(Set("Running", "Reconnecting").contains(phase)))
    and (phase == "Suspended" implies sleeping)
  run cancellationTest = init.then(take("startOk")).then(take("sleep")).then(take("stop"))
    .then(take("wake")).then(take("retryOk")).expect(phase == "Selected" and not(requested))
  run exhaustedTest = init.then(take("startTransient")).then(take("retryErr")).then(take("retryErr"))
    .then(take("retryErr")).then(take("retryErr")).then(take("retryErr")).then(take("wait"))
    .expect(phase == "BackOff" and attempts == 5)
}
