module confidence {
  var event: str
  var score: str
  var good: int
  var bad: int
  var confident: bool
  var signal: str
  action init = all {
    event' = "reset", score' = "invalid", good' = 0, bad' = 0, confident' = false, signal' = "None",
  }
  action take(op: str, input: str): bool = {
    val high = Set("enter", "aboveEnter").contains(input)
    val low = Set("exit", "belowExit", "invalid").contains(input)
    val countGood = if (low) 0 else if (high and good < 3) good + 1 else good
    val countBad = if (high) 0 else if (low and bad < 3) bad + 1 else bad
    val acquire = op == "new" and not(confident) and countGood >= 2
    val degrade = op == "new" and confident and countBad >= 3
    val clear = op == "reset" or acquire or degrade
    all {
      event' = op, score' = input,
      good' = if (clear) 0 else if (op == "new") countGood else good,
      bad' = if (clear) 0 else if (op == "new") countBad else bad,
      confident' = (acquire or (confident and not(degrade) and op != "reset")),
      signal' = if (acquire) "Acquire" else if (degrade) "Degrade" else "None",
    }
  }
  action step = {
    nondet op = Set("new", "held", "reset").oneOf()
    nondet input = Set("belowExit", "exit", "aboveExit", "belowEnter", "enter", "aboveEnter", "invalid").oneOf()
    take(op, input)
  }
  val safety = and {
    good == 0 or bad == 0, good <= 3, bad <= 3,
    not(confident) implies good < 2, confident implies bad < 3,
    signal != "None" implies (event == "new" and good == 0 and bad == 0),
  }
  run hysteresisTest = init.then(take("new", "enter")).then(take("held", "enter"))
    .then(take("new", "belowEnter")).expect(good == 1 and not(confident))
    .then(take("new", "aboveEnter")).expect(signal == "Acquire")
    .then(take("new", "exit")).then(take("held", "exit")).then(take("new", "aboveExit"))
    .then(take("new", "invalid")).then(take("new", "belowExit")).expect(signal == "Degrade")
  run saturationTest = init.then(take("new", "exit")).then(take("new", "exit"))
    .then(take("new", "exit")).then(take("new", "exit")).expect(bad == 3)
}
