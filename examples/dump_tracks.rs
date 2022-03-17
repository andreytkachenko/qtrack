use grant_tracker::meter::scene::{Participant, Scene};
use grant_common::Detection;
use nalgebra as na;

fn map_to_top_view(det: &Detection, offset: (f32, f32), height: i32, fov: f32, angle: f32, scale: f32) -> Detection {
    let appy = fov / height as f32;
    let h2 = height as f32 / 2.0;
    let y = (height as f32 - det.y) - h2;
    let ay = y * appy;
    let s = (angle + ay).tan();
    println!("{} {}", y, s);

    Detection {
        x: offset.0 + (det.x * s) * scale,
        y: offset.1 + ((-y) * s) * scale,
        w: (det.w * s) * scale,
        h: (det.h * s) * scale,
        confidence: det.confidence,
        class: det.class,
    }
}

fn main() -> Result<(), grant_tracker::error::Error> {
    use std::io::BufRead;

    let mut args = std::env::args();

    let _ = args.next().unwrap();
    let in_file_name = args.next().expect("expected detections file name");
    let dets_file = std::fs::File::open(in_file_name)?;


    let width  = 1280.0f32;
    let height = 720.0f32;

    let padding = 20.0;
    let poly = vec![
        nalgebra::Point2::new(padding, padding),
        nalgebra::Point2::new(width - padding - padding, padding),
        nalgebra::Point2::new(width - padding - padding, height - padding - padding),
        nalgebra::Point2::new(padding, height - padding - padding),
    ];

    let mut scene = Scene::new(poly);

    let reader = std::io::BufReader::new(dets_file).lines();
    let mut line_reader = reader.into_iter();

    loop {
        let (ts, xx): (f64, Vec<Detection>) = match line_reader.next() {
            Some(x) => {
                let x = x.unwrap();

                if let Some(idx) = x.find(':') {
                    let (ts, vector) = x.split_at(idx);

                    match (ts.parse::<u64>(), serde_json::from_str(&vector[1..])) {
                        (Ok(ts), Ok(vector)) => (ts as f64, vector),
                        (Ok(_), _) => {
                            eprintln!("wrong file format: parse json failed");
                            continue
                        },
                        (_, Ok(_)) => {
                            eprintln!("wrong file format: parse timestamp failed");
                            continue
                        },
                        _ => {
                            eprintln!("wrong file format: parse failed");
                            continue
                        },
                    }
                } else {
                    eprintln!("wrong file format: expected `:`");
                    continue;
                }
            },
            None => break,
        };

        let ts_us = (ts * 1000.0).round() as u64;
        let mapping = scene.map_detections(ts_us, &xx);
        scene.update(mapping);

        for t in &scene.tracks {
            if t.id > 0 {
                println!("{} {} {} {} {}", t.id, t.object.position.x, t.object.position.y, t.last_hit_score, ts_us);
            }
        }
    }

    Ok(())
}