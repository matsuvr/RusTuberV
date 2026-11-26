// Detector and landmark ROI lifecycle at an interval of three source frames.
module inference {
  var roi: str
  var lastDetector: int
  var seq: int
  var event: str
  var ran: bool
  var valid: bool
  action init = all {
    roi' = "Empty", lastDetector' = -1, seq' = 0, event' = "init", ran' = false, valid' = true,
  }
  action frame(id: int, outcome: str): bool = {
    val detect = roi != "Tracking" or (lastDetector >= 0 and id - lastDetector >= 3)
    val tracked = outcome == "good"
    all {
      seq' = id, event' = outcome, ran' = detect, lastDetector' = if (detect) id else lastDetector,
      roi' = if (tracked) "Tracking" else "Lost",
      valid' = and {
        roi != "Tracking" implies detect,
        (roi == "Tracking" and id < lastDetector + 3) implies not(detect),
        (id >= lastDetector + 3) implies detect,
      },
    }
  }
  action step = {
    nondet id = 0.to(5).oneOf()
    nondet outcome = Set("good", "low", "outside").oneOf()
    frame(id, outcome)
  }
  val safety = valid and (roi == "Tracking" implies (lastDetector >= 0 and event == "good"))
    and (event == "low" or event == "outside" implies roi == "Lost")
  run redetectTest = init.then(frame(1, "good")).then(frame(2, "low")).expect(not(ran) and roi == "Lost")
    .then(frame(2, "good")).expect(ran).then(frame(4, "good")).expect(not(ran))
    .then(frame(5, "outside")).expect(ran and roi == "Lost")
}
