use anyhow::{Result, bail};

/// Download a raw camera frame (JPEG bytes). No image decoding.
pub fn fetch_frame(client: &reqwest::blocking::Client, url: &str, token: &str) -> Result<Vec<u8>> {
    let resp = client.get(url).header("Authorization", token).send()?.error_for_status()?;
    let bytes = resp.bytes()?.to_vec();
    if bytes.len() < 2 || bytes[0] != 0xff || bytes[1] != 0xd8 {
        bail!("response from {url} is not a JPEG (got {} bytes)", bytes.len());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAMERAS: [(&str, &str); 2] = [
        ("http://192.168.0.118:50051/frame", "983149"),
        ("http://192.168.0.107:50051/frame", "378031"),
    ];

    /// Hits a live camera on the LAN and writes `frame.jpg` next to this crate.
    /// Run with:
    /// `cargo test -p camera-fetch download_camera_frame -- --ignored --nocapture`
    #[test]
    #[ignore = "requires live camera on LAN"]
    fn download_camera_frame() -> Result<()> {
        let (url, token) = CAMERAS[0];
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()?;

        let bytes = fetch_frame(&client, url, token)?;
        assert!(!bytes.is_empty(), "downloaded frame from {url} should not be empty");

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("frame.jpg");
        std::fs::write(&path, &bytes)?;
        println!("wrote {} bytes from {url} to {}", bytes.len(), path.display());
        Ok(())
    }
}
