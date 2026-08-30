# Model Manifest for Acceptance Testing

This manifest lists VRM 1.0 models used in acceptance testing.
Each model must have a verified SHA-256 hash.

## Models

| # | Name | Source | License | SHA-256 | VRM Version | Expressions | Notes |
|---|------|--------|---------|---------|-------------|-----------|-------|
| 1 | _fill in_ | _url_ | _license_ | _sha256_ | 1.0 | _list_ | _notes_ |
| 2 | _fill in_ | _url_ | _license_ | _sha256_ | 1.0 | _list_ | _notes_ |
| 3 | _fill in_ | _url_ | _license_ | _sha256_ | 1.0 | _list_ | _notes_ |

## Verification

To verify a model hash:

```bash
sha256sum <model-file.vrm>
# or on Windows:
certutil -hashfile <model-file.vrm> SHA256
```

Compare the output with the SHA-256 column above.

## Skip Conditions

- Model is not VRM 1.0 → SKIP (not a failure, but not tested)
- Model hash does not match → FAIL (do not use unverified models)
- Model missing required bones (hips, head) → FAIL at preflight
- Model has no expressions → capability limitation, not failure

## Camera Devices

| # | Device Name | Backend | Resolution | Format | Notes |
|---|------------|---------|-----------|--------|-------|
| 1 | _fill in_ | MSMF | _WxH_ | _format_ | _notes_ |
| 2 | _fill in_ | MSMF | _WxH_ | _format_ | _if available_ |
