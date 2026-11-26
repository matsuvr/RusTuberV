// A replacement includes the unloading update and a newly ready generation.
module manual_expression {
  var event: str
  var selected: str
  var hasCatalog: bool
  var writer: str
  action init = all { event' = "init", selected' = "", hasCatalog' = true, writer' = "" }
  pure def toggle(old: str, id: str): str = if (old == id) "" else id
  action take(op: str): bool = {
    val choice = if (op == "replace" or op == "clear") "" else if (Set("happy","JawOpen","unknown").contains(op)) toggle(selected,op)
      else if (op == "pair") toggle(toggle(selected,"happy"),"JawOpen") else selected
    val catalog = if (op == "dropCatalog") false else if (op == "restore" or op == "replace") true else hasCatalog
    all {
      event' = op, selected' = choice, hasCatalog' = catalog,
      writer' = if (catalog and Set("happy","JawOpen").contains(choice)) choice else "",
    }
  }
  action step = {
    nondet op = Set("replace","clear","happy","JawOpen","unknown","pair","staleClear","staleToggle","dropCatalog","restore").oneOf()
    take(op)
  }
  val safety = and { writer != "" implies (hasCatalog and writer == selected),
    event == "replace" implies selected == "", selected == "unknown" implies writer == "", }
  run staleTest = init.then(take("happy")).then(take("replace")).then(take("staleToggle")).expect(writer == "")
    .then(take("pair")).then(take("staleClear")).expect(writer == "JawOpen").then(take("dropCatalog"))
    .expect(writer == "" and selected == "JawOpen").then(take("restore")).then(take("JawOpen")).expect(writer == "")
}
