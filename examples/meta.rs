use std::sync::Arc;
use opencv::{
    core::{self, Mat},
    highgui,
    prelude::*,
    videoio,
};
use nalgebra as na;

fn solve_linear(a: &na::Matrix2<f32>, b: &na::Vector2<f32>) -> Option<na::Vector2<f32>> {
    let qr_result = a.qr();
    let qty = qr_result.q().transpose() * b;
    let beta_hat = qr_result.r().solve_upper_triangular(&qty);

    beta_hat
}

fn line(x1: f32, y1: f32, x2: f32, y2: f32) -> (f32, f32, f32) {
    (
        y1 - y2,
        x2 - x1,
        x1 * y2 - x2 * y1,
    )
}

fn dist(angle: f32, dist: f32, app: f32) -> impl Fn(f32) -> f32 {
    move |x| {
        
        dist / ((angle + x * app).tan())
    }
}


fn main() -> Result<(), grant_tracker::error::Error> {
    let f = dist(std::f32::consts::FRAC_PI_4, 1.0, std::f32::consts::FRAC_PI_2 / 720.0);
    println!("{}", f(0.0));
    println!("{}", f(-10.0));
    println!("{}", f(-20.0));
    println!("{}", f(-350.0));

    let mut args = std::env::args();
    let _ = args.next().unwrap();
    let in_file_name = args.next().expect("expected video file name");
    let drawing = Arc::new(std::sync::Mutex::new(None));
    let drawing_clone = drawing.clone();

    let window = "video capture";
    highgui::named_window(window, 1)?;
    highgui::set_mouse_callback(window, Some(Box::new(move |a, b, c, d| match a {
        1 => {
            *drawing_clone.lock().unwrap() = Some((false, b, c, b, c));
        },
        0 => if let Some(l) = &mut *drawing_clone.lock().unwrap() {
            l.3 = b;
            l.4 = c;
        },
        4 => {
            if let Some(l) = &mut *drawing_clone.lock().unwrap() {
                l.0 = true;
                l.3 = b;
                l.4 = c;
            }
        },
        _ => ()
    })));

    let mut cam = videoio::VideoCapture::from_file(&in_file_name, videoio::CAP_ANY)?;  // 0 is the default camera
   
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    let mut frame = core::Mat::new_rows_cols_with_default(height, width,  core::CV_8UC3, core::Scalar::new(0.0, 0.0, 0.0, 0.0)).unwrap();
    let mut video_frame = core::Mat::new_rows_cols_with_default(height, width,  core::CV_8UC3, core::Scalar::new(0.0, 0.0, 0.0, 0.0)).unwrap();

    let mut frame_idx = 0;

    let mut paused = false;
    let mut lines = vec![];

    loop {
        if !paused {
            frame_idx += 1;

            if cam.read(&mut video_frame).is_err() {
                break;
            }
        
            let (fwidth, fheight) = (frame.cols(), frame.rows()); 

            if fwidth == 0 || fheight == 0 {
                break;
            }

            video_frame.copy_to(&mut frame)?;

            let offset = cam.get(videoio::CAP_PROP_POS_MSEC)?;

            opencv::imgproc::put_text(
                &mut frame, 
                &format!("{}", frame_idx), 
                core::Point::new(10 as i32, 30 as i32), 
                opencv::imgproc::FONT_HERSHEY_SIMPLEX, 
                1.0, 
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                opencv::imgproc::LINE_AA, 
                false
            )?;
        } else {
            video_frame.copy_to(&mut frame)?; 
        }

        let copy = drawing.lock().unwrap().clone();

        if let Some((f, x1, y1, x2, y2)) = copy {
            if f {
                println!("a {}, b {}, c {}", y1 - y2, x2 - x1, x1 * y2 - x2 * y1);

                lines.push((x1, y1, x2, y2));

                let last = lines.len() - 1;
                if lines.len() % 2 == 0 && lines.len() > 1 {
                    let (a1, b1, c1) = line(lines[last - 1].0 as _, lines[last - 1].1 as _, lines[last - 1].2 as _, lines[last - 1].3 as _);
                    let (a2, b2, c2) = line(lines[last].0 as _, lines[last].1 as _, lines[last].2 as _, lines[last].3 as _);
                    
                    let a = na::Matrix2::new(
                        a1, b1,
                        a2, b2,
                    );
                    let b = na::Vector2::new(
                        -c1,
                        -c2,
                    );
                    let res = solve_linear(&a, &b).unwrap();
                    println!("{}", res.y);
                }

                *drawing.lock().unwrap() = None;
            } else {
                opencv::imgproc::line(
                    &mut frame, 
                    core::Point::new(
                        x1 as i32, 
                        y1 as i32,
                    ), 
                    core::Point::new(
                        x2 as i32, 
                        y2 as i32, 
                    ),
                    core::Scalar::new(255.0, 255.0, 0.0, 0.0), 
                    1, 
                    opencv::imgproc::LINE_8, 
                    0
                )?;
            }
        }

        for &(x1, y1, x2, y2) in &lines {
            opencv::imgproc::line(
                &mut frame, 
                core::Point::new(
                    x1 as i32, 
                    y1 as i32,
                ), 
                core::Point::new(
                    x2 as i32, 
                    y2 as i32, 
                ),
                core::Scalar::new(0.0, 255.0, 0.0, 0.0), 
                1, 
                opencv::imgproc::LINE_8, 
                0
            )?;
        }

        highgui::imshow(window, &mut frame)?;

        let key = highgui::wait_key(20)?;

        const K: i32 = b'u' as i32;

        match key {
            32 => paused = !paused,
            27 => break,
            K => { lines.pop(); },
            _ => ()
        }
    }


    Ok(())
}