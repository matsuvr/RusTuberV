// Clearance and joint margin are distances from the two numerical acceptance
// thresholds used by Problem::feasible. 0 is exactly on the boundary.
module arm_admission {
  var clearance: int
  var joint: int
  var candidate: int
  var displayed: int
  var admitted: bool
  var retained: bool
  action init = all {
    clearance' = -1, joint' = -1, candidate' = 0, displayed' = 0, admitted' = false, retained' = true,
  }
  action sample(c: int, j: int, pose: int): bool = {
    val ok = c >= 0 and j >= 0
    val output = if (ok) pose else displayed
    all {
      clearance' = c, joint' = j, candidate' = pose, admitted' = ok, displayed' = output,
      retained' = ((c < 0 or j < 0) implies output == displayed),
    }
  }
  action step = {
    nondet c = (-1).to(1).oneOf()
    nondet j = (-1).to(1).oneOf()
    nondet pose = Set(1, 2).oneOf()
    sample(c, j, pose)
  }
  val safety = retained and (admitted implies (displayed == candidate and clearance >= 0 and joint >= 0))
  run boundaryTest = init.then(sample(0, 0, 1)).expect(admitted)
    .then(sample(-1, 1, 2)).expect(not(admitted) and displayed == 1)
    .then(sample(1, -1, 2)).expect(not(admitted) and displayed == 1)
}
