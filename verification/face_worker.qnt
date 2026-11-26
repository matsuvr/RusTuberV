// MediaPipe frame processing: a successful no-face result breaks an error
// streak. The eleventh consecutive recoverable error stops this worker.
module face_worker {
  var streak: int
  var failed: bool
  var closed: bool
  var published: bool
  var rejected: bool
  var event: str
  action init = all {
    streak' = 0, failed' = false, closed' = false, published' = false, rejected' = false, event' = "init",
  }
  action take(op: str): bool = {
    val count = if (op == "error") streak + 1 else if (op == "noFace") 0 else streak
    all {
      not(failed) or op == "close",
      event' = op, streak' = count, failed' = (count > 10),
      closed' = (op == "close" or closed),
      published' = (op == "noFace" and not(closed)), rejected' = (op == "noFace" and closed),
    }
  }
  action step = { nondet op = Set("noFace", "error", "close").oneOf() take(op) }
  val safety = and {
    failed == (streak == 11),
    event == "noFace" implies (streak == 0 and not(failed) and (published != rejected)),
    published implies not(closed),
    event == "error" implies not(published or rejected),
  }
  run thresholdTest = init.then(take("error")).then(take("noFace")).expect(streak == 0)
    .then(take("error")).then(take("error")).then(take("error")).then(take("error")).then(take("error"))
    .then(take("error")).then(take("error")).then(take("error")).then(take("error")).then(take("error"))
    .expect(not(failed)).then(take("error")).expect(failed)
}
