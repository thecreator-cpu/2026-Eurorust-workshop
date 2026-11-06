use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use opencv::{
    core::{Mat, Point2f, Size, Vector},
    imgcodecs, imgproc,
    objdetect::{self, ArucoDetector, DetectorParameters, PredefinedDictionaryType, RefineParameters},
    prelude::*,
};

const CAR_ID: i32 = 5;

const CAMERAS: [(&str, &str); 2] = [
    ("http://192.168.0.118:50051/frame", "983149"),
    ("http://192.168.0.107:50051/frame", "378031"),
];

/// Arena corner marker IDs in TL, TR, BR, BL order, as laid out in camera 11's view.
/// Must match the quadrant orientation the `/goal` endpoint and dashboard use.
const CORNER_IDS: [i32; 4] = [14, 13, 0, 12];

#[derive(Debug, Clone, Copy)]
struct Pose {
    /// Arena coordinates: (0, 0) is top-left, (1, 1) is bottom-right.
    x: f64,
    y: f64,
    /// Radians, 0 points along +x, increasing towards +y.
    heading: f64,
}

impl Pose {
    fn quadrant(&self) -> &'static str {
        match (self.x < 0.5, self.y < 0.5) {
            (true, true) => "TL",
            (false, true) => "TR",
            (false, false) => "BR",
            (true, false) => "BL",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Move {
    /// Shortest relative turn in radians: + = CCW, − = CW, range (−π, π].
    turn_rad: f64,
    /// Euclidean distance in arena units.
    distance: f64,
}

/// Center of a goal quadrant in arena coordinates.
fn quadrant_center(q: &str) -> Option<(f64, f64)> {
    match q {
        "TL" => Some((0.25, 0.25)),
        "TR" => Some((0.75, 0.25)),
        "BR" => Some((0.75, 0.75)),
        "BL" => Some((0.25, 0.75)),
        _ => None,
    }
}

/// Wrap angle into (−π, π].
fn wrap_to_pi(a: f64) -> f64 {
    use std::f64::consts::PI;
    (a + PI).rem_euclid(2.0 * PI) - PI
}

/// Distance and shortest signed turn from `pose` toward `(target_x, target_y)`.
fn next_move(pose: &Pose, target_x: f64, target_y: f64) -> Move {
    let dx = target_x - pose.x;
    let dy = target_y - pose.y;
    let distance = dx.hypot(dy);
    if distance < 1e-9 {
        return Move { turn_rad: 0.0, distance: 0.0 };
    }
    let bearing = dy.atan2(dx);
    Move {
        turn_rad: wrap_to_pi(bearing - pose.heading),
        distance,
    }
}

fn fetch_frame(client: &reqwest::blocking::Client, url: &str, token: &str) -> Result<Mat> {
    let resp = client.get(url).header("Authorization", token).send()?.error_for_status()?;
    let bytes = resp.bytes()?;
    let buf = Vector::<u8>::from_slice(&bytes);
    let img = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        bail!("could not decode JPEG from {url}");
    }
    Ok(img)
}

fn make_detector() -> Result<ArucoDetector> {
    let dictionary = objdetect::get_predefined_dictionary(PredefinedDictionaryType::DICT_4X4_50)?;
    let mut params = DetectorParameters::default()?;
    // Markers are small in 1080p; default min perimeter (3% of max side ≈ 58px) drops them.
    params.set_min_marker_perimeter_rate(0.005);
    params.set_min_corner_distance_rate(0.01);
    params.set_min_marker_distance_rate(0.01);
    params.set_polygonal_approx_accuracy_rate(0.05);
    params.set_adaptive_thresh_constant(7.0);
    params.set_adaptive_thresh_win_size_min(3);
    params.set_adaptive_thresh_win_size_max(23);
    params.set_adaptive_thresh_win_size_step(4);
    // Keep ECC moderate — higher rates invent phantom IDs (17, 37, …) on foreshortened tags.
    params.set_error_correction_rate(0.6);
    params.set_use_aruco3_detection(false);
    let refine = RefineParameters::new_def()?;
    Ok(ArucoDetector::new(&dictionary, &params, refine)?)
}

/// Markers are only ~30-50px and heavily foreshortened, so each scale decodes a different subset.
const DETECT_SCALES: [f64; 4] = [1.0, 2.0, 3.0, 4.0];

fn detect_on_gray(
    detector: &ArucoDetector,
    gray: &Mat,
    ox: f32,
    oy: f32,
    scales: &[f64],
    out: &mut HashMap<i32, [Point2f; 4]>,
) -> Result<()> {
    for &scale in scales {
        let mut scaled = Mat::default();
        imgproc::resize(gray, &mut scaled, Size::default(), scale, scale, imgproc::INTER_CUBIC)?;

        let mut corners = Vector::<Mat>::new();
        let mut ids = Vector::<i32>::new();
        let mut rejected = Vector::<Mat>::new();
        detector.detect_markers(&scaled, &mut corners, &mut ids, &mut rejected)?;

        let s = scale as f32;
        for (id, quad) in ids.iter().zip(corners.iter()) {
            let q = quad.data_typed::<Point2f>()?;
            out.entry(id).or_insert([
                Point2f::new(q[0].x / s + ox, q[0].y / s + oy),
                Point2f::new(q[1].x / s + ox, q[1].y / s + oy),
                Point2f::new(q[2].x / s + ox, q[2].y / s + oy),
                Point2f::new(q[3].x / s + ox, q[3].y / s + oy),
            ]);
        }
    }
    Ok(())
}

fn corner_hits(markers: &HashMap<i32, [Point2f; 4]>) -> usize {
    CORNER_IDS.iter().filter(|id| markers.contains_key(id)).count()
}

/// Returns each detected marker's four corners (marker TL, TR, BR, BL) keyed by ID,
/// in original image pixel coordinates.
fn detect(detector: &ArucoDetector, img: &Mat) -> Result<HashMap<i32, [Point2f; 4]>> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(img, &mut gray, imgproc::COLOR_BGR2GRAY)?;

    // CLAHE helps low-contrast / foreshortened markers on the wood floor.
    let mut clahe = imgproc::create_clahe(2.0, Size::new(8, 8))?;
    let mut enhanced = Mat::default();
    clahe.apply(&gray, &mut enhanced)?;

    let mut out = HashMap::new();
    // Full frame + overlapping tiles: small far markers often only decode in a tile
    // where they are a larger fraction of the image.
    for gray_src in [&gray, &enhanced] {
        detect_on_gray(detector, gray_src, 0.0, 0.0, &DETECT_SCALES, &mut out)?;
        let w = gray_src.cols();
        let h = gray_src.rows();
        let tw = w * 2 / 3;
        let th = h * 2 / 3;
        for (x0, y0) in [(0, 0), (w - tw, 0), (0, h - th), (w - tw, h - th)] {
            let roi = Mat::roi(gray_src, opencv::core::Rect::new(x0, y0, tw, th))?.try_clone()?;
            detect_on_gray(detector, &roi, x0 as f32, y0 as f32, &[2.0, 3.0, 4.0], &mut out)?;
        }
        if corner_hits(&out) >= 3 && out.contains_key(&CAR_ID) {
            break;
        }
    }
    Ok(out)
}

fn center(q: &[Point2f; 4]) -> Point2f {
    let sx: f32 = q.iter().map(|p| p.x).sum();
    let sy: f32 = q.iter().map(|p| p.y).sum();
    Point2f::new(sx / 4.0, sy / 4.0)
}

fn perspective(h: &Mat, pts: &[Point2f]) -> Result<Vec<Point2f>> {
    let src = Vector::<Point2f>::from_slice(pts);
    let mut dst = Vector::<Point2f>::new();
    opencv::core::perspective_transform(&src, &mut dst, h)?;
    Ok(dst.to_vec())
}

fn locate(markers: &HashMap<i32, [Point2f; 4]>, corner_ids: &[i32; 4]) -> Result<Option<Pose>> {
    let Some(car) = markers.get(&CAR_ID) else {
        return Ok(None);
    };

    let mut pts: [Option<Point2f>; 4] = corner_ids.map(|id| markers.get(&id).map(center));
    let missing: Vec<usize> = (0..4).filter(|&i| pts[i].is_none()).collect();
    let mut seen: Vec<_> = markers.keys().copied().collect();
    seen.sort();
    println!("markers {seen:?}");
    match missing.as_slice() {
        [] => {}
        // Approximate a single hidden corner by completing the parallelogram from the opposite one.
        &[i] => {
            let (a, b, opp) = (pts[(i + 1) % 4].unwrap(), pts[(i + 3) % 4].unwrap(), pts[(i + 2) % 4].unwrap());
            pts[i] = Some(Point2f::new(a.x + b.x - opp.x, a.y + b.y - opp.y));
        }
        _ => {
            let ids: Vec<i32> = missing.iter().map(|&i| corner_ids[i]).collect();
            bail!("corner markers {ids:?} not visible");
        }
    }
    let src = Vector::<Point2f>::from_iter(pts.into_iter().map(Option::unwrap));
    let dst = Vector::<Point2f>::from_slice(&[
        Point2f::new(0.0, 0.0),
        Point2f::new(1.0, 0.0),
        Point2f::new(1.0, 1.0),
        Point2f::new(0.0, 1.0),
    ]);
    let h = opencv::geometry::get_perspective_transform(&src, &dst, opencv::core::DECOMP_LU)?;

    let c = center(car);
    let front = Point2f::new((car[0].x + car[1].x) / 2.0, (car[0].y + car[1].y) / 2.0);
    let mapped = perspective(&h, &[c, front])?;
    let (p, f) = (mapped[0], mapped[1]);

    Ok(Some(Pose {
        x: p.x as f64,
        y: p.y as f64,
        heading: ((f.y - p.y) as f64).atan2((f.x - p.x) as f64),
    }))
}

fn fuse_poses(poses: &[Pose]) -> Option<Pose> {
    if poses.is_empty() {
        return None;
    }
    let n = poses.len() as f64;
    let x = poses.iter().map(|p| p.x).sum::<f64>() / n;
    let y = poses.iter().map(|p| p.y).sum::<f64>() / n;
    let (s, c) = poses
        .iter()
        .fold((0.0, 0.0), |(s, c), p| (s + p.heading.sin(), c + p.heading.cos()));
    Some(Pose { x, y, heading: s.atan2(c) })
}

/// Fetch live camera frames, locate the car, and plan a move toward `goal` quadrant.
fn live_next_move(
    client: &reqwest::blocking::Client,
    detector: &ArucoDetector,
    goal: &str,
) -> Result<(Pose, Move)> {
    let mut poses = Vec::new();
    for (url, token) in CAMERAS {
        match fetch_frame(client, url, token).and_then(|img| detect(detector, &img)) {
            Ok(markers) => {
                let mut seen: Vec<_> = markers.keys().copied().collect();
                seen.sort();
                let corners: Vec<_> = CORNER_IDS
                    .iter()
                    .copied()
                    .filter(|id| markers.contains_key(id))
                    .collect();
                eprintln!("camera {url}: markers {seen:?} (corners {corners:?})");
                for &id in &CORNER_IDS {
                    match markers.get(&id) {
                        Some(q) => {
                            let c = center(q);
                            eprintln!("  corner {id:>2} at pixel ({:.0}, {:.0})", c.x, c.y);
                        }
                        None => eprintln!("  corner {id:>2} not detected"),
                    }
                }
                match markers.get(&CAR_ID) {
                    Some(q) => {
                        let c = center(q);
                        eprintln!("  car {CAR_ID} at pixel ({:.0}, {:.0})", c.x, c.y);
                    }
                    None => eprintln!("  car {CAR_ID} not among detected markers"),
                }
                match locate(&markers, &CORNER_IDS) {
                    Ok(Some(pose)) => {
                        eprintln!(
                            "  pose x={:.3} y={:.3} heading={:.0}°",
                            pose.x,
                            pose.y,
                            pose.heading.to_degrees()
                        );
                        poses.push(pose);
                    }
                    Ok(None) => {}
                    Err(e) => eprintln!("  locate: {e:#}"),
                }
            }
            Err(e) => eprintln!("camera {url}: {e:#}"),
        }
    }
    let pose = fuse_poses(&poses).with_context(|| {
        format!("no usable pose for car {CAR_ID} (need car + ≥3 arena corners on one camera)")
    })?;
    let (tx, ty) = quadrant_center(goal).with_context(|| format!("unknown goal quadrant {goal}"))?;
    Ok((pose, next_move(&pose, tx, ty)))
}

fn main() -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()?;
    let detector = make_detector()?;

    // Optional image paths (one per camera) to run on saved frames instead of fetching.
    let files: Vec<String> = std::env::args().skip(1).collect();

    let mut poses = Vec::new();
    for (i, (url, token)) in CAMERAS.iter().enumerate() {
        let frame = if files.is_empty() {
            fetch_frame(&client, url, token)
        } else if let Some(path) = files.get(i) {
            imgcodecs::imread(path, imgcodecs::IMREAD_COLOR).map_err(Into::into)
        } else {
            continue;
        };
        let url = files.get(i).map_or(*url, String::as_str);
        let result = frame
            .and_then(|img| detect(&detector, &img))
            .and_then(|markers| {
                let mut seen: Vec<_> = markers.keys().copied().collect();
                seen.sort();
                println!("camera {url}: markers {seen:?}");
                for id in seen {
                    let c = center(&markers[&id]);
                    println!("  marker {id:>2} at pixel ({:.0}, {:.0})", c.x, c.y);
                }
                locate(&markers, &CORNER_IDS)
            });
        match result {
            Ok(Some(pose)) => {
                println!(
                    "  car {CAR_ID}: x={:.3} y={:.3} heading={:.0}° quadrant={}",
                    pose.x,
                    pose.y,
                    pose.heading.to_degrees(),
                    pose.quadrant()
                );
                poses.push(pose);
            }
            Ok(None) => println!("  car {CAR_ID} not visible"),
            Err(e) => println!("  error: {e:#}"),
        }
    }

    let fused = fuse_poses(&poses).with_context(|| format!("car {CAR_ID} not found on any camera"))?;
    println!(
        "car {CAR_ID}: x={:.3} y={:.3} heading={:.0}° quadrant={}",
        fused.x,
        fused.y,
        fused.heading.to_degrees(),
        fused.quadrant()
    );

    // Hard-coded goal until HTTP `/goal` wiring lands.
    const GOAL: &str = "TR";
    let (tx, ty) = quadrant_center(GOAL).expect("valid goal quadrant");
    let mv = next_move(&fused, tx, ty);
    println!(
        "next move → {GOAL} ({tx:.2}, {ty:.2}): turn={:.0}° distance={:.3}",
        mv.turn_rad.to_degrees(),
        mv.distance
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrant_centers() {
        assert_eq!(quadrant_center("TL"), Some((0.25, 0.25)));
        assert_eq!(quadrant_center("TR"), Some((0.75, 0.25)));
        assert_eq!(quadrant_center("BR"), Some((0.75, 0.75)));
        assert_eq!(quadrant_center("BL"), Some((0.25, 0.75)));
        assert_eq!(quadrant_center("XX"), None);
    }

    #[test]
    fn next_move_straight_ahead() {
        let pose = Pose { x: 0.0, y: 0.5, heading: 0.0 };
        let mv = next_move(&pose, 1.0, 0.5);
        assert!((mv.turn_rad).abs() < 1e-9);
        assert!((mv.distance - 1.0).abs() < 1e-9);
    }

    #[test]
    fn next_move_shortest_turn_is_clockwise() {
        // Heading −170°, target bearing +170° → shortest turn is −20° (CW).
        let pose = Pose {
            x: 0.5,
            y: 0.5,
            heading: (-170f64).to_radians(),
        };
        let tx = 0.5 + 170f64.to_radians().cos();
        let ty = 0.5 + 170f64.to_radians().sin();
        let mv = next_move(&pose, tx, ty);
        assert!((mv.turn_rad.to_degrees() - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn next_move_on_target_is_zero() {
        let pose = Pose { x: 0.25, y: 0.25, heading: 1.23 };
        let mv = next_move(&pose, 0.25, 0.25);
        assert_eq!(mv, Move { turn_rad: 0.0, distance: 0.0 });
    }

    /// Generate reference DICT_4X4_50 corner markers. Run with:
    /// `cargo test generate_corner_markers -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic"]
    fn generate_corner_markers() -> Result<()> {
        let dictionary = objdetect::get_predefined_dictionary(PredefinedDictionaryType::DICT_4X4_50)?;
        for id in [0, 12, 13, 14] {
            let mut marker = Mat::default();
            objdetect::generate_image_marker(&dictionary, id, 200, &mut marker, 1)?;
            let path = format!("/tmp/aruco4x4_50_{id}.png");
            imgcodecs::imwrite(&path, &marker, &Vector::new())?;
            println!("wrote {path}");
        }
        Ok(())
    }

    /// Hits a live camera on the LAN. Run with:
    /// `cargo test download_camera_frame -- --ignored --nocapture`
    #[test]
    #[ignore = "requires live camera on LAN"]
    fn download_camera_frame() -> Result<()> {
        let (url, token) = CAMERAS[0];
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()?;

        let img = fetch_frame(&client, url, token)?;
        assert!(!img.empty(), "decoded frame from {url} should not be empty");
        assert!(img.rows() > 0 && img.cols() > 0, "frame should have nonzero size");
        println!("downloaded {}x{} frame from {url}", img.cols(), img.rows());
        Ok(())
    }

    /// Poll both cameras each loop until all 4 arena corners are visible on each.
    /// Prints every attempt. Run with:
    /// `cargo test poll_until_all_corners -- --ignored --nocapture`
    #[test]
    #[ignore = "requires live cameras on LAN"]
    fn poll_until_all_corners() -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()?;
        let detector = make_detector()?;

        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let mut all_done = true;
            for &(url, token) in &CAMERAS {
                match fetch_frame(&client, url, token).and_then(|img| {
                    println!(
                        "attempt {attempt} camera {url}: frame {}x{}",
                        img.cols(),
                        img.rows()
                    );
                    detect(&detector, &img)
                }) {
                    Ok(markers) => {
                        let mut seen: Vec<_> = markers.keys().copied().collect();
                        seen.sort();
                        let hits = corner_hits(&markers);
                        println!("  markers {seen:?} ({hits}/4 corners)");
                        for &id in &CORNER_IDS {
                            match markers.get(&id) {
                                Some(q) => {
                                    let c = center(q);
                                    println!("  corner {id:>2} at pixel ({:.0}, {:.0})", c.x, c.y);
                                }
                                None => println!("  corner {id:>2} not detected"),
                            }
                        }
                        if hits == CORNER_IDS.len() {
                            println!("  camera {url}: all 4 corners found");
                        } else {
                            all_done = false;
                        }
                    }
                    Err(e) => {
                        println!("attempt {attempt} camera {url}: {e:#}");
                        all_done = false;
                    }
                }
            }
            if all_done {
                println!("both cameras have all 4 corners after {attempt} attempt(s)");
                break;
            }
        }
        Ok(())
    }

    /// Full pipeline: fetch both cameras → detect → locate → next move toward TR.
    /// Run with:
    /// `cargo test live_next_move_to_goal -- --ignored --nocapture`
    #[test]
    #[ignore = "requires live cameras on LAN"]
    fn live_next_move_to_goal() -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()?;
        let detector = make_detector()?;

        const GOAL: &str = "TR";
        let (pose, mv) = live_next_move(&client, &detector, GOAL)?;

        assert!((0.0..=1.0).contains(&pose.x), "x={} out of arena", pose.x);
        assert!((0.0..=1.0).contains(&pose.y), "y={} out of arena", pose.y);
        assert!(mv.distance.is_finite() && mv.distance >= 0.0);
        assert!(mv.turn_rad.abs() <= std::f64::consts::PI + 1e-9);

        let (tx, ty) = quadrant_center(GOAL).unwrap();
        println!(
            "pose x={:.3} y={:.3} heading={:.0}° quadrant={}",
            pose.x,
            pose.y,
            pose.heading.to_degrees(),
            pose.quadrant()
        );
        println!(
            "next move → {GOAL} ({tx:.2}, {ty:.2}): turn={:.0}° distance={:.3}",
            mv.turn_rad.to_degrees(),
            mv.distance
        );
        Ok(())
    }
}
