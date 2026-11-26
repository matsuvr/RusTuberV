// Face and arm workers finish asynchronously. Each new session has its own time origin.
module observation_age {
  var event: str
  var nowNs: int
  var slotSeq: int
  var slotTime: int
  var lastFace: int
  var dispatch: str
  var heldSeq: int
  var heldTime: int
  action init = all {
    event' = "init", nowNs' = 0, slotSeq' = 0, slotTime' = 0, lastFace' = 0,
    dispatch' = "missing", heldSeq' = 0, heldTime' = 0,
  }
  action resetSession = all {
    event' = "reset", nowNs' = 0, slotSeq' = 0, slotTime' = 0, lastFace' = 0,
    dispatch' = "missing", heldSeq' = 0, heldTime' = 0,
  }
  action readFrame(op: str, seq: int, captured: int, clock: int): bool = {
    val publish = op == "publish"
    val id = if (publish) seq else slotSeq
    val stamp = if (publish) captured else slotTime
    val unavailable = id == 0 or stamp < 0 or clock - stamp > 250000000
    val answer = if (unavailable) "missing" else if (id == lastFace) "held" else "new"
    // Arms retain the last admitted observation if an older session's frame arrives.
    val admitArm = publish and captured >= 0
    val armId = if (admitArm) seq else heldSeq
    val armTime = if (admitArm) captured else heldTime
    val usable = armId != 0 and clock - armTime <= 250000000
    all {
      clock >= nowNs, publish implies captured <= clock,
      event' = op, nowNs' = clock, slotSeq' = id, slotTime' = stamp,
      lastFace' = if (answer == "new") id else lastFace, dispatch' = answer,
      heldSeq' = if (usable) armId else 0, heldTime' = if (usable) armTime else 0,
    }
  }
  action step = any { resetSession, {
    nondet clock = Set(0,1,249999999,250000000,250000001,250000002).oneOf()
    any { readFrame("tick",0,0,clock), {
      nondet id = Set(1,2).oneOf()
      nondet stamp = (-1).to(1).oneOf()
      readFrame("publish",id,stamp,clock)
    } }
  } }
  val safety = and {
    dispatch != "missing" implies (slotTime >= 0 and nowNs - slotTime <= 250000000 and lastFace == slotSeq),
    heldSeq != 0 implies (heldTime >= 0 and nowNs - heldTime <= 250000000),
  }
  run deadlineTest = init.then(readFrame("publish",1,0,0))
    .then(readFrame("tick",0,0,250000000)).expect(dispatch == "held" and heldSeq == 1)
    .then(readFrame("tick",0,0,250000001)).expect(dispatch == "missing" and heldSeq == 0)
    .then(resetSession).then(readFrame("publish",1,-1,0)).expect(dispatch == "missing")
    .then(readFrame("publish",1,0,0)).expect(dispatch == "new")
  run lateWorkerTest = init.then(readFrame("publish",2,1,1)).then(readFrame("publish",1,-1,1))
    .expect(dispatch == "missing" and heldSeq == 2)
    .then(readFrame("tick",0,0,250000002)).expect(heldSeq == 0)
}
