// Camera lists are encoded as the stable IDs in dropdown order.
module camera_selection {
  var event: str
  var devices: str
  var selected: str
  var phase: str
  var errorKind: str
  var desired: bool
  var ack: bool
  action init = all {
    event' = "init", devices' = "AB", selected' = "", phase' = "Idle",
    errorKind' = "", desired' = false, ack' = true,
  }
  action take(op: str): bool = {
    val refresh = Set("AB", "BA", "B", "empty").contains(op)
    val pick = Set("first", "second", "invalid").contains(op)
    val id = if (op == "first" and devices != "empty") (if (devices == "AB") "A" else "B")
      else if (op == "second" and devices == "AB") "B"
      else if (op == "second" and devices == "BA") "A" else ""
    val change = pick and id != "" and (id != selected or phase == "Failed")
    val recover = refresh and errorKind == "camera"
    val stop = (change or (recover and phase == "Failed")) and phase != "Idle" and phase != "Stopping"
    val failed = op == "cameraError" or op == "inferenceError"
    val retry = op == "retry" and errorKind == "inference"
    val start = (op == "auto" and phase == "Idle" or retry) and selected != ""
    all {
      op == "started" implies phase == "Starting",
      op == "stopped" implies phase == "Stopping",
      failed implies (phase == "Starting" or phase == "Running"),
      event' = op, devices' = if (refresh) op else devices,
      selected' = if (change) id else if (op == "empty" or (op == "B" and selected == "A")) "" else selected,
      errorKind' = if (op == "cameraError") "camera" else if (op == "inferenceError") "inference"
        else if (recover or change or retry) "" else errorKind,
      phase' = if (failed or stop) "Stopping" else if (start) "Starting"
        else if (op == "started") "Running" else if (op == "stopped") (if (errorKind == "") "Idle" else "Failed")
        else phase,
      desired' = if (failed or stop) false else if (start) true else desired,
      ack' = if (failed or stop or start) false else if (op == "started" or op == "stopped") true else ack,
    }
  }
  action step = {
    nondet op = Set("AB","BA","B","empty","first","second","invalid","cameraError","inferenceError",
      "auto","autoWithoutAvatar","started","stopped","retry").oneOf()
    take(op)
  }
  val safety = and {
    desired == Set("Running","Starting").contains(phase),
    devices == "empty" implies selected == "", devices == "B" implies selected != "A",
    phase == "Running" implies ack,
  }
  run changeTest = init.then(take("first")).then(take("autoWithoutAvatar")).expect(phase == "Idle")
    .then(take("auto")).then(take("started")).then(take("BA")).then(take("second")).expect(phase == "Running" and selected == "A")
    .then(take("first")).then(take("auto")).expect(phase == "Stopping")
    .then(take("stopped")).then(take("auto")).expect(phase == "Starting" and selected == "B")
  run recoveryTest = init.then(take("second")).then(take("auto")).then(take("cameraError")).then(take("stopped"))
    .then(take("retry")).expect(phase == "Failed").then(take("B")).expect(phase == "Stopping")
    .then(take("stopped")).then(take("auto")).then(take("inferenceError")).then(take("stopped"))
    .then(take("retry")).expect(phase == "Starting" and desired and not(ack))
}
