use opencv::{
    core,
    highgui,
};

use std::{collections::{BTreeMap, HashMap}, io::BufRead};
use nalgebra as na;
use meter::tracker::Object;

fn main() -> Result<(), anyhow::Error> {
    let color = core::Scalar::new(0.0, 0.0, 255.0, 0.0);
    let window = "video capture";
    highgui::named_window(window, 1)?;
    let mut frame = core::Mat::new_rows_cols_with_default(1024, 1400,  core::CV_8UC3, core::Scalar::new(255.0, 255.0, 255.0, 0.0)).unwrap();

    let file = std::fs::File::open("/home/andrey/track_01.dat")?;
    let reader = std::io::BufReader::new(file)
        .lines()
        .into_iter()
        .filter_map(|x| {
            let x = x.ok()?;
            let mut i = x.split(' ');
            let id: i32 = i.next().unwrap().parse().ok()?;
            let center_x: f32 = i.next().unwrap().parse().ok()?;
            let center_y: f32 = i.next().unwrap().parse().ok()?;
            let conf: f32 = i.next().and_then(|x|x.parse().ok()).unwrap_or(1.0);
            let ts: u64 = i.next().and_then(|x|x.parse().ok()).unwrap_or(0);
            let ts = ts as f32 / 1000_000.0;

            Some((id, (na::Point2::new(center_x , center_y), ts, conf)))
        });

    let mut data = BTreeMap::new();
    for (id, value) in reader {
        data.entry(id)
            .or_insert_with(Vec::new)
            .push(value);
    }

    for (key, data) in data {
        if !([5].contains(&key)) {
            continue;
        }

        print!("K{}: ", key);

        let mut reader = data.into_iter();
        let mut pts: Vec<(na::Point2<f32>, f32, na::Complex<f32>)> = Vec::new();
        let mut predictions: Vec<(na::Point2<f32>, f32)> = Vec::new();

        let p = reader.next().unwrap();
        let mut obj = Object::new(meter::tracker::BodyKind::Car, 25.0, 1.0, p.1, p.0, p.2);

        let mut e_cnt = 0.0;
        let mut e_err = 0.0;
        let mut e_err_min = 1.0;

        for (pp, ts, w) in reader {
            let d3 = obj.probability(ts, pp, w);
            e_err_min = if d3 < e_err_min {
                d3 
            } else {
                e_err_min
            };

            e_err += if d3.is_nan() { 0.0 } else {d3};
            e_cnt += 1.0;

            predictions.push((pp, d3));

            if let Some((_, pp)) = obj.update(ts, pp, w) {
                pts.push((pp, obj.velocity, obj.direction))
            }
        }

        println!("err: {}; {}", e_err_min, 1.0 - e_err / e_cnt);
   
        for &(pos, vel, dir) in &pts {
            opencv::imgproc::circle(
                &mut frame, 
                core::Point::new(
                    pos.x as i32,
                    pos.y as i32,
                ), 
                4, 
                color, 
                opencv::imgproc::FILLED, 
                opencv::imgproc::LINE_AA, 
                0
            )?;

            let offset = na::Complex::new(0.0, -15.0) * dir;

            opencv::imgproc::line(
                &mut frame, 
                core::Point::new(
                    (pos.x + offset.re) as i32,
                    (pos.y + offset.im) as i32,
                ), 
                core::Point::new(
                    (pos.x - offset.re) as i32,
                    (pos.y - offset.im) as i32,
                ), 
                color, 
                1, 
                opencv::imgproc::LINE_AA, 
                0
            )?;

            opencv::imgproc::put_text(
                &mut frame, 
                &format!("{:0.0}", vel / 10.0), 
                core::Point::new(
                    (pos.x + offset.re) as i32,
                    (pos.y + offset.im) as i32,
                ), 
                opencv::imgproc::FONT_HERSHEY_SIMPLEX, 
                0.2, 
                core::Scalar::new(0.0, 0.0, 0.0, 255.0),
                1,
                opencv::imgproc::LINE_AA, 
                false
            )?;
        }

        for (p, conf) in &predictions {
            opencv::imgproc::circle(
                &mut frame, 
                core::Point::new(
                    p.x as i32,
                    p.y as i32,
                ), 
                1, 
                // core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                core::Scalar::new((( conf) * 255.0) as f64, ((conf) * 255.0) as f64, ((conf) * 255.0) as f64, 0.0), 
                opencv::imgproc::FILLED, 
                opencv::imgproc::LINE_8, 
                0
            )?;
        }
    }
    loop {
        highgui::imshow(window, &frame)?;

        let key = highgui::wait_key(100)?;
        if key == 27 {
            break;
        }
    }

    Ok(())
}