use std::io::{Write, stdout};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode},
};

const CAR_URL: &str = "http://192.168.0.197:5000";
const CAR_TOKEN: &str = "987993";
const PULSE_MS: u64 = 300;
const SPEED: f64 = 0.5;

#[derive(Clone, Copy)]
enum Action {
    Forward,
    RotateLeft,
    RotateRight,
}

impl Action {
    fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::RotateLeft => "rotate left",
            Self::RotateRight => "rotate right",
        }
    }

    fn command(self) -> (f64, bool) {
        match self {
            Self::Forward => (SPEED, false),
            Self::RotateLeft => (SPEED, true),
            Self::RotateRight => (-SPEED, true),
        }
    }
}

struct RawMode;

impl RawMode {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enable raw mode")?;
        Ok(Self)
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

fn put_command(client: &reqwest::blocking::Client, speed: f64, flip: bool) -> Result<()> {
    let body = format!(r#"{{"speed":{speed},"flip":{flip}}}"#);
    client
        .put(CAR_URL)
        .header("Authorization", CAR_TOKEN)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .and_then(|r| r.error_for_status())
        .with_context(|| format!("PUT {CAR_URL} speed={speed} flip={flip}"))?;
    Ok(())
}

fn pulse(client: &reqwest::blocking::Client, action: Action) -> Result<()> {
    let (speed, flip) = action.command();
    put_command(client, speed, flip)?;
    std::thread::sleep(Duration::from_millis(PULSE_MS));
    put_command(client, 0.0, false)?;
    Ok(())
}

fn map_key(code: KeyCode) -> Option<Action> {
    match code {
        KeyCode::Char('w') | KeyCode::Up => Some(Action::Forward),
        KeyCode::Char('a') | KeyCode::Left => Some(Action::RotateLeft),
        KeyCode::Char('d') | KeyCode::Right => Some(Action::RotateRight),
        _ => None,
    }
}

fn is_quit(code: KeyCode) -> bool {
    matches!(code, KeyCode::Char('q') | KeyCode::Esc)
}

fn main() -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;

    let _raw = RawMode::enter()?;
    let mut out = stdout();
    writeln!(
        out,
        "drive car 5 — w/↑ forward, a/← left, d/→ right, q/Esc quit\r"
    )?;
    out.flush()?;

    loop {
        let Event::Key(KeyEvent { code, kind, .. }) = event::read()? else {
            continue;
        };
        // Crossterm on some platforms emits Press+Release; act once per Press.
        if kind != KeyEventKind::Press {
            continue;
        }
        if is_quit(code) {
            break;
        }
        let Some(action) = map_key(code) else {
            continue;
        };
        match pulse(&client, action) {
            Ok(()) => writeln!(out, "{}\r", action.label())?,
            Err(e) => writeln!(out, "error: {e:#}\r")?,
        }
        out.flush()?;
    }

    if let Err(e) = put_command(&client, 0.0, false) {
        writeln!(out, "stop failed: {e:#}\r")?;
    }
    writeln!(out, "stopped\r")?;
    Ok(())
}
