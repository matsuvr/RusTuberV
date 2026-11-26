// LatestSlot, with two independent consumers and up to three publications.
module slot {
  var state: { generation: int, value: int, closed: bool, replacements: int, face: int, pose: int }
  var event: str
  var result: str
  var valid: bool

  action init = all {
    state' = { generation: 0, value: 0, closed: false, replacements: 0, face: 0, pose: 0 },
    event' = "init", result' = "none", valid' = true,
  }

  action take(op: str): bool = {
    val publish = op == "publishA" or op == "publishB"
    val read = op == "face" or op == "pose"
    val cursor = if (op == "face") state.face else state.pose
    val fresh = state.value != 0 and state.generation > cursor
    val outcome =
      if (publish) { if (state.closed) "rejected" else "published" }
      else if (read) { if (state.closed) "closed" else if (fresh) "new" else "none" }
      else "none"
    val after =
      if (outcome == "published") {
        { ...state, generation: state.generation + 1, value: if (op == "publishA") 1 else 2,
          replacements: state.replacements + (if (state.value == 0) 0 else 1) }
      } else if (op == "clear" or op == "close") {
        { ...state, value: 0, closed: state.closed or op == "close" }
      } else if (outcome == "new" and op == "face") { { ...state, face: state.generation } }
      else if (outcome == "new") { { ...state, pose: state.generation } }
      else state
    all {
      outcome != "published" or state.generation < 3,
      state' = after, event' = op, result' = outcome,
      valid' = and {
        state.closed implies after == state,
        read implies (after.value == state.value and after.generation == state.generation),
        op == "face" implies after.pose == state.pose,
        op == "pose" implies after.face == state.face,
        (op == "clear" or op == "close") implies (after.value == 0 and after.generation == state.generation),
        outcome == "new" implies (not(state.closed) and after.generation > cursor),
      },
    }
  }
  action step = {
    nondet op = Set("publishA", "publishB", "face", "pose", "clear", "close").oneOf()
    take(op)
  }
  val safety = valid and state.face <= state.generation and state.pose <= state.generation
    and (state.closed implies state.value == 0) and state.replacements <= state.generation

  run consumersTest = init.then(take("publishA")).then(take("pose")).then(take("publishB"))
    .then(take("face")).then(take("pose")).then(take("clear")).then(take("publishA"))
    .then(take("close")).then(take("face")).then(take("publishB"))
    .expect(result == "rejected" and state.face == 2 and state.pose == 2 and state.replacements == 1)
}
