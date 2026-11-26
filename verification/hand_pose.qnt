module hand_pose {
  var event: str
  var selected: str
  var candidate: str
  var frames: int
  var strong: int
  action init = all {
    event' = "missing", selected' = "", candidate' = "", frames' = 0, strong' = 0,
  }
  action observe(op: str): bool = {
    val winner = if (op == "open") "open" else if (op == "missing") "" else "fist"
    val fresh = winner != "" and winner != selected
    val n = if (winner == candidate) frames + 1 else 1
    val s = if (op == "ambiguous") 0 else if (winner == candidate) strong + 1 else 1
    val confirmed = fresh and (s >= 2 or n >= 4)
    val pending = fresh and not(confirmed)
    all {
      event' = op, selected' = if (confirmed) winner else selected,
      candidate' = if (pending) winner else "", frames' = if (pending) n else 0,
      strong' = if (pending) s else 0,
    }
  }
  action step = { nondet op = Set("open","fist","ambiguous","missing").oneOf() observe(op) }
  val safety = and {
    frames < 4, strong < 2, strong <= frames,
    (frames == 0) == (candidate == ""), candidate != "" implies candidate != selected,
    event == "missing" implies frames == 0,
  }
  run interruptionTest = init.then(observe("open")).then(observe("open")).expect(selected == "open")
    .then(observe("fist")).then(observe("missing")).then(observe("fist")).expect(selected == "open")
    .then(observe("ambiguous")).then(observe("fist")).expect(selected == "open")
    .then(observe("ambiguous")).expect(selected == "fist")
}
