// Two observation instants per collection. Batch sizes exercise the real 15/38 limits.
module auto_neutral {
  var event: str
  var batch: int
  var gapNs: int
  var phase: str
  var recent: int
  var robust: bool
  action init = all {
    event' = "reset", batch' = 0, gapNs' = 0, phase' = "empty", recent' = 0, robust' = false,
  }
  action take(op: str, n: int, gap: int): bool = {
    val kept = if (gap > 1200000000) 0 else recent
    val total = kept + n
    all {
      op == "seed" implies phase == "empty",
      op == "later" implies phase == "early",
      event' = op, batch' = n, gapNs' = gap,
      phase' = if (op == "reset") "empty" else if (op == "later") "late"
        else if (op == "seed" or op == "recenter") "early" else phase,
      recent' = if (op == "reset") 0 else if (op == "recenter") 1
        else if (op == "seed" or op == "later") (if (total > 38) 38 else total) else recent,
      robust' = if (op == "later") (kept > 0 and gap >= 1120000000 and total >= 15)
        else if (op == "invalid" or op == "invalidRecenter") robust else false,
    }
  }
  action step = any {
    { nondet n = Set(1, 13, 14, 15, 38, 39).oneOf() take("seed", n, 0) },
    { nondet gap = Set(1119999999,1120000000,1120000001,1199999999,1200000000,1200000001).oneOf()
      nondet n = Set(1, 14).oneOf() take("later", n, gap) },
    { nondet op = Set("invalid","invalidRecenter","reset","recenter").oneOf() take(op, 0, 0) },
  }
  val safety = (phase == "empty") == (recent == 0) and recent <= 38 and
    (robust implies (phase == "late" and recent >= 15))
  run limitsTest = init.then(take("seed",14,0)).then(take("later",1,1120000000)).expect(robust)
    .then(take("invalidRecenter",0,0)).expect(robust).then(take("recenter",0,0)).expect(recent == 1 and not(robust))
    .then(take("later",14,1200000000)).expect(robust).then(take("reset",0,0))
    .then(take("seed",39,0)).expect(recent == 38).then(take("later",14,1200000001)).expect(recent == 14 and not(robust))
  run belowTest = init.then(take("seed",14,0)).then(take("later",1,1119999999)).expect(not(robust))
    .then(take("reset",0,0)).then(take("seed",13,0)).then(take("later",1,1120000000)).expect(not(robust))
}
