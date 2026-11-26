module calibration_samples {
  var event: str
  var seqKind: str
  var stamp: int
  var quality: str
  var count: int
  var lastStamp: int
  var decision: str
  action clear = all {
    event' = "reset", seqKind' = "same", stamp' = 0, quality' = "threshold",
    count' = 0, lastStamp' = -1, decision' = "reset",
  }
  action init = all {
    event' = "init", seqKind' = "same", stamp' = 0, quality' = "threshold",
    count' = 0, lastStamp' = -1, decision' = "reset",
  }
  action offer(order: str, t: int, q: str): bool = {
    val verdict = if (order != "fresh") "duplicate" else if (t < lastStamp) "regressed"
      else if (count == 5) "full" else if (q == "invalid") "invalid" else if (q == "below") "low" else "accepted"
    all {
      event' = "offer", seqKind' = order, stamp' = t, quality' = q, decision' = verdict,
      count' = if (verdict == "accepted") count + 1 else count,
      lastStamp' = if (verdict == "accepted") t else lastStamp,
    }
  }
  action step = any { clear, {
    nondet order = Set("fresh","same","older").oneOf()
    nondet t = 0.to(1).oneOf()
    nondet q = Set("threshold","below","invalid").oneOf()
    offer(order,t,q)
  } }
  val safety = count >= 0 and count <= 5 and ((count == 0) == (lastStamp == -1)) and
    (decision == "accepted" implies (seqKind == "fresh" and stamp == lastStamp and quality == "threshold"))
  run highWaterTest = init.then(offer("fresh",1,"below")).then(offer("same",0,"threshold")).expect(decision == "duplicate")
    .then(offer("fresh",0,"threshold")).expect(count == 1).then(offer("fresh",1,"threshold"))
    .then(offer("fresh",0,"threshold")).expect(decision == "regressed")
  run capacityTest = init.then(offer("fresh",1,"threshold")).then(offer("fresh",1,"threshold"))
    .then(offer("fresh",1,"threshold")).then(offer("fresh",1,"threshold")).then(offer("fresh",1,"threshold"))
    .then(offer("fresh",1,"invalid")).expect(decision == "full").then(offer("fresh",0,"threshold")).expect(decision == "regressed")
    .then(clear).expect(count == 0)
}
