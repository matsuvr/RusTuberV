// Begin at the channel boundary, with discovery pending for physical display A.
module bridge_discovery {
  var event: str
  var loading: bool
  var device: str
  var saving: str
  var failed: bool
  action init = all { event' = "init", loading' = true, device' = "", saving' = "", failed' = false }
  action take(op: str): bool = {
    val cancel = Set("stop","cancelReady","hld","hldReady").contains(op)
    val result = loading and not(cancel) and op != "wait"
    all {
      event' = op, loading' = (loading and not(result) and not(cancel)),
      device' = if (result) (if (op == "A" or op == "B") op else "") else device,
      saving' = if (op == "hld" or op == "hldReady") "HLD" else if (cancel) ""
        else if (result and op == "A") "A" else saving,
      failed' = if (cancel) false else if (result) op != "A" else failed,
    }
  }
  action step = {
    nondet op = Set("wait","stop","cancelReady","hld","hldReady","A","B","empty","malformed","disconnected","error").oneOf()
    take(op)
  }
  val safety = and { loading implies (saving == "" and not(failed)),
    saving == "A" implies (device == "A" and not(failed)),
    Set("stop","cancelReady").contains(event) implies (saving == "" and not(loading)),
  }
  run cancelTest = init.then(take("wait")).then(take("cancelReady")).then(take("A"))
    .expect(not(loading) and saving == "" and device == "")
  run replaceTest = init.then(take("hldReady")).then(take("error")).expect(saving == "HLD" and not(failed))
  run missingTest = init.then(take("B")).expect(failed and device == "B" and saving == "")
  run successTest = init.then(take("A")).expect(not(failed) and saving == "A").then(take("stop")).expect(saving == "")
}
