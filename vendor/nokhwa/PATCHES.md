Upstream: nokhwa 0.10.11 from crates.io (Apache-2.0).

The AVFoundation frame and raw-frame reads use a two-second receive timeout
instead of an unbounded wait. A stopped camera must release the capture worker
so sleep recovery and shutdown can proceed. Remove this patch when upstream
provides bounded AVFoundation reads.
