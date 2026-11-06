# Manual drive UI (tap pulses)

## Goal

Smallest usable control loop so a human can point Car 5 and nudge it forward by tapping keys in the terminal. No reverse. Vision / goal-seeking stays separate.

## Approach

New workspace crate `drive/` (`drive/src/main.rs`). Run with `cargo run -p drive`. Does not link OpenCV or touch `car-tracker` / `main.rs`.

## Car target

Hardcoded (from team card for Car 5 / marker id 5):

- URL: `http://192.168.0.197:5000`
- Auth: header `Authorization: 987993`
- Method: `PUT` root with JSON `{"speed": <f64>, "flip": <bool>}`

Respect car constraints from the hackathon README: ~100ms cooldown between requests; commands expire after 1s.

## Controls

Raw terminal input (crossterm). Each keypress triggers one pulse; keys pressed during an in-flight pulse are ignored.

| Key | Action | Command |
|-----|--------|---------|
| `w` or ↑ | forward | `{"speed": 0.5, "flip": false}` |
| `a` or ← | rotate left | `{"speed": 0.5, "flip": true}` |
| `d` or → | rotate right | `{"speed": -0.5, "flip": true}` |
| `q` or Esc | quit | send stop, restore terminal, exit |

No reverse (`s` / ↓ ignored or unused).

Rotate direction sign may need a one-line flip after first live test if left/right feel swapped.

## Pulse timing

1. `PUT` the motion command.
2. Sleep ~300ms.
3. `PUT` stop: `{"speed": 0.0, "flip": false}`.

Print a one-line status after each successful pulse (e.g. `forward`, `rotate left`). On HTTP error, print the error and keep the loop running so the next tap can retry.

On quit (and ideally on panic unwind via raw-mode restore), always leave the car stopped.

## Dependencies

- `reqwest` (blocking) — already in `car-tracker`
- `crossterm` — keyboard / raw mode
- JSON body: tiny hand-built string or `serde_json` (whichever is less ceremony)

## Out of scope

- Hold-to-drive / continuous key repeat
- Reverse
- Camera pose / goal integration
- Multi-car selection CLI
- Web UI
