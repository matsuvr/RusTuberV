module video_frame {
  var width: int
  var height: int
  var strideClass: str
  var sizeDelta: int
  var result: str
  var packedLength: int
  action init = all { width' = 0, height' = 0, strideClass' = "packed", sizeDelta' = 0,
    result' = "ZeroDimensions", packedLength' = 0 }
  action readback(w: int, h: int, stride: str, delta: int): bool = {
    val error = if (w == 0) "ZeroDimensions" else if (stride == "short") "InvalidStride"
      else if (h == 0) "ZeroDimensions" else if (stride == "max" and h > 1) "SizeOverflow"
      else if (stride == "max" or delta != 0) "DataLength" else "ok"
    all {
      width' = w, height' = h, strideClass' = stride, sizeDelta' = delta, result' = error,
      packedLength' = if (error == "ok") w * h * 4 else 0,
    }
  }
  action step = {
    nondet w = 0.to(2).oneOf()
    nondet h = 0.to(2).oneOf()
    nondet stride = Set("short","packed","padded","max").oneOf()
    nondet delta = (-1).to(1).oneOf()
    readback(w,h,stride,delta)
  }
  val safety = (result == "ok") == (width > 0 and height > 0 and
    Set("packed","padded").contains(strideClass) and sizeDelta == 0) and (result == "ok") == (packedLength > 0)
  run paddingTest = init.then(readback(2,2,"padded",0)).expect(packedLength == 16)
    .then(readback(2,2,"short",0)).expect(result == "InvalidStride")
    .then(readback(2,2,"max",0)).expect(result == "SizeOverflow")
    .then(readback(2,1,"max",0)).expect(result == "DataLength")
    .then(readback(2,2,"packed",-1)).expect(result == "DataLength")
    .then(readback(2,2,"padded",1)).expect(result == "DataLength")
}
