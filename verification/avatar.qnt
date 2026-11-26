// Two root identities (0 means absent). Generations 1..3 permit repeated
// replacement of the same root while an older instance is still unloading.
module avatar {
  type State = { phase: str, root: int, pending: int, generation: int }
  var state: State
  var event: { op: str, root: int }
  var accepted: bool
  var valid: bool
  var binding: { root: int, generation: int }
  var frameGeneration: int
  var drives: bool
  action init = all {
    state' = { phase: "NoAvatar", root: 0, pending: 0, generation: 0 },
    event' = { op: "init", root: 0 }, accepted' = false, valid' = true,
    binding' = { root: 0, generation: 0 }, frameGeneration' = 0, drives' = false,
  }
  action take(op: str, root: int): bool = {
    val empty = state.phase == "NoAvatar" or state.phase == "Failed"
    val load = (op == "load" or op == "replace") and empty
    val replace = op == "replace" and Set("Ready", "Unloading").contains(state.phase)
    val unload = op == "unload" and state.phase == "Ready"
    val bind = op == "bind" and state.phase == "Loading" and root == state.root
    val finish = op == "finish" and state.phase == "Unloading"
    val after =
      if (load) { phase: "Loading", root: root, pending: 0, generation: state.generation + 1 }
      else if (replace) { ...state, phase: "Unloading", pending: root, generation: state.generation + 1 }
      else if (unload) { ...state, phase: "Unloading", pending: 0 }
      else if (finish) { ...state, phase: if (state.pending == 0) "NoAvatar" else "Loading", root: state.pending, pending: 0 }
      else if (bind) { ...state, phase: "Binding" }
      else if (op == "ready" and state.phase == "Binding") { ...state, phase: "Ready" }
      else if (op == "fail") { ...state, phase: "Failed", root: 0, pending: 0 }
      else state
    val saved = if (op == "saveBinding" and state.phase == "Ready")
      { root: state.root, generation: state.generation } else binding
    val frame = if (op == "saveFrame" and not(empty)) state.generation else frameGeneration
    all {
      after.generation <= 3,
      state' = after, event' = { op: op, root: root }, accepted' = (load or replace or unload),
      binding' = saved, frameGeneration' = frame,
      drives' = (after.phase == "Ready" and saved.root == after.root
                 and saved.generation == after.generation and frame == after.generation),
      valid' = and {
        (after.phase == "Ready" and state.phase != "Ready") implies state.phase == "Binding",
        (op == "bind" and root != state.root) implies after == state,
        finish implies (after.root == state.pending and after.generation == state.generation),
        replace implies (after.root == state.root and after.pending == root and after.generation > state.generation),
        (Set("load", "replace", "unload").contains(op) and not(load or replace or unload)) implies after == state,
      },
    }
  }
  action step = any {
    { nondet op = Set("load", "replace", "bind").oneOf()
      nondet root = Set(1, 2).oneOf()
      take(op, root) },
    { nondet op = Set("unload", "finish", "ready", "fail", "saveBinding", "saveFrame").oneOf()
      take(op, 0) },
  }
  val safety = valid and ((state.root == 0) == Set("NoAvatar", "Failed").contains(state.phase))
    and (state.pending != 0 implies state.phase == "Unloading")
    and (state.root != 0 implies state.generation > 0)
    and (drives implies (state.phase == "Ready" and binding.generation == state.generation
                         and binding.root == state.root and frameGeneration == state.generation))
  run coalesceTest = init.then(take("load", 1)).then(take("bind", 2)).then(take("ready", 0))
    .then(take("bind", 1)).then(take("ready", 0)).then(take("replace", 2)).then(take("replace", 1))
    .then(take("finish", 0)).expect(state.phase == "Loading" and state.root == 1 and state.generation == 3)
  run retryTest = init.then(take("replace", 2)).then(take("fail", 0)).then(take("load", 1))
    .then(take("bind", 1)).then(take("ready", 0)).then(take("unload", 0)).then(take("finish", 0))
    .expect(state.phase == "NoAvatar" and state.root == 0)
  run staleFrameTest = init.then(take("load", 1)).then(take("bind", 1)).then(take("ready", 0))
    .then(take("saveBinding", 0)).then(take("saveFrame", 0)).expect(drives)
    .then(take("replace", 1)).then(take("finish", 0)).then(take("bind", 1)).then(take("ready", 0))
    .expect(not(drives)).then(take("saveBinding", 0)).expect(not(drives))
    .then(take("saveFrame", 0)).expect(drives)
}
