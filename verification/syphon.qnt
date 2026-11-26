// A step covers one native client poll/send. Producers act during its callback.
module syphon {
  var event: str
  var connected: bool
  var clients: bool
  var status: str
  var pending: int
  var offered: int
  action init = all {
    event' = "none", connected' = false, clients' = false, status' = "Starting", pending' = 0, offered' = 0,
  }
  action poll(op: str, found: bool): bool = {
    val stopped = status == "Off" or op == "stop"
    val consume = if (status == "Off") 0 else pending
    val keep = clients and found and not(stopped)
    all {
      event' = op, connected' = found, offered' = consume, clients' = (found and not(stopped)),
      pending' = if (keep and op == "submit") 1 else if (keep and op == "replace") 2 else 0,
      status' = if (stopped) "Off" else if (not(found)) "Starting" else if (consume != 0) "Live" else status,
    }
  }
  action step = {
    nondet op = Set("none","submit","replace","discard","stop").oneOf()
    nondet found = Set(false,true).oneOf()
    poll(op,found)
  }
  val safety = and { not(clients) implies pending == 0, status == "Live" implies clients,
    event == "discard" implies pending == 0, status == "Off" implies not(clients), event == "stop" implies status == "Off", }
  run latePollTest = init.then(poll("submit",true)).expect(pending == 0)
    .then(poll("replace",true)).expect(pending == 2)
    .then(poll("submit",false)).expect(offered == 2 and pending == 0 and status == "Starting")
    .then(poll("none",true)).then(poll("submit",true)).then(poll("none",true)).expect(status == "Live")
    .then(poll("discard",true)).then(poll("stop",true)).expect(status == "Off" and not(clients))
}
