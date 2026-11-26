module expression_keys {
  var event: str
  var key: int
  var expression: str
  var first: str
  var second: str
  pure def binding(old: str, slot: int, op: str, k: int, id: str): str =
    if (op == "defaults") (if (slot == 1) "a" else "b")
    else if (op == "assign" and k == slot) id
    else if (op == "clear" or (op == "unassign" and k == slot)
      or (Set("assign","remove").contains(op) and id == old)) "" else old
  action init = all { event' = "clear", key' = 1, expression' = "a", first' = "", second' = "" }
  action take(op: str, k: int, id: str): bool = all {
    event' = op, key' = k, expression' = id,
    first' = binding(first,1,op,k,id), second' = binding(second,2,op,k,id),
  }
  action step = {
    nondet op = Set("defaults","assign","unassign","remove","clear").oneOf()
    nondet k = Set(1,2).oneOf()
    nondet id = Set("a","b").oneOf()
    take(op,k,id)
  }
  val safety = (first == "" or first != second) and
    (event == "assign" implies (if (key == 1) first else second) == expression)
  run moveTest = init.then(take("defaults",1,"a")).then(take("assign",2,"a")).expect(first == "" and second == "a")
    .then(take("remove",1,"b")).expect(second == "a").then(take("remove",1,"a")).expect(second == "")
}
