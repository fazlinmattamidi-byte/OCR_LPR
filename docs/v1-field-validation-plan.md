# V1 Field Validation Plan

This plan is for proving the browser ANPR pipeline with real vehicles before a Version 1.0 release.

## Performance Targets

| Metric | Target |
| --- | --- |
| Detector median latency | < 80 ms |
| Detector P95 latency | < 250 ms |
| OCR median latency | < 150 ms |
| OCR P95 latency | < 500 ms |
| Sustained detector FPS | >= 5 |
| Track loss rate | < 5% |
| Duplicate OCR rate | < 1% |
| False exact matches | 0 |
| Memory growth over 30 minutes | < 10% |

## Scenario Matrix

| Scenario | Status | Notes |
| --- | --- | --- |
| Stationary camera | [ ] | Baseline accuracy and latency |
| Moving camera | [ ] | Handheld and vehicle-mounted |
| Parked vehicle | [ ] | Near and far plates |
| Moving vehicle | [ ] | Slow and moderate speed |
| Camera and vehicle both moving | [ ] | Mobile ANPR target case |
| Rain | [ ] | Wet plates and reflections |
| Night | [ ] | Low-light and headlight glare |
| Backlight | [ ] | Sun behind vehicle |
| Motorcycle | [ ] | Small/two-line plates |
| Truck | [ ] | High/low plate positions |
| Dirty plate | [ ] | Low contrast |
| Reflective plate | [ ] | Glare handling |
| Two adjacent vehicles | [ ] | Track separation |
| Heavy traffic | [ ] | OCR queue pressure |
| 30-minute run | [ ] | Memory, temperature, latency drift |
| 1-hour run | [ ] | Stability and reconnect behavior |
| 2-hour run | [ ] | Extended thermal throttling |

## Recovery Tests

| Failure Mode | Expected Behavior | Status |
| --- | --- | --- |
| Camera permission revoked | Scanner stops cleanly and shows camera error | [ ] |
| Camera disconnected | Stream stops, runtime state survives, preview can restart | [ ] |
| Browser tab backgrounded | Detection pauses/slows without stale OCR alerts | [ ] |
| Device locked/unlocked | Camera can be restarted without stale tracks | [ ] |
| Thermal throttling | Device tier/adaptive pacing prevents queue runaway | [ ] |
| WebGPU/WASM reset | Runtime banner reports degraded/unavailable state | [ ] |
| Low-memory pressure | Metrics export shows memory growth and dropped frames | [ ] |

## Field Accuracy Report Template

| Scenario | Vehicles | Detection Recall | Detector Precision | OCR Full-Plate Accuracy | OCR Character Accuracy | Consensus Success | DB Match Accuracy | False Positives | False Negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Day | 0 |  |  |  |  |  |  |  |  |
| Night | 0 |  |  |  |  |  |  |  |  |
| Rain | 0 |  |  |  |  |  |  |  |  |
| Highway | 0 |  |  |  |  |  |  |  |  |
| City | 0 |  |  |  |  |  |  |  |  |

## V1 Release Gate

| Gate | Target | Status |
| --- | --- | --- |
| Critical runtime failures | 0 during validation runs | [ ] |
| False exact database alerts | 0 | [ ] |
| Full-plate recognition rate | Meets field target agreed from validation data | [ ] |
| P95 response time | Within detector and OCR targets above | [ ] |
| Memory growth | No uncontrolled growth over 1-2 hours | [ ] |
| Long-duration stability | Stable operation for at least 1-2 hours | [ ] |
| Device coverage | Representative Android and iPhone devices pass | [ ] |

## Release-Readiness Issues

| Issue | Why It Matters | Status |
| --- | --- | --- |
| Duplicate `FULL UI DESIGN` app tree | Parallel app trees can drift and increase release risk | [ ] |
| Root lint debt outside scanner | Public release should clear remaining app-wide lint/compiler findings | [ ] |
| Existing `StorageContext` React compiler lint errors | Build passes, but public release should remove React compiler incompatibilities | [ ] |
| Long-duration memory data | Required before claiming 30-minute and 1-hour stability | [ ] |
