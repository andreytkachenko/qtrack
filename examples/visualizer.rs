use grant_common::Detection;
use opencv::{
    core::{self, Mat},
    highgui,
    prelude::*,
    videoio,
};
use nalgebra as na;

fn draw_pred(frame: &mut Mat, det: &Detection) -> opencv::Result<()> {
    let rect = core::Rect::new(
        (det.x - 2.0) as i32, 
        (det.y - 2.0) as i32, 
        4 as i32, 
        4 as i32
    );

    //  Draw a bounding box.
    opencv::imgproc::rectangle(
        frame, 
        rect, 
        core::Scalar::new(0.0, 255.0, 255.0, 128.0), 
        1, 
        opencv::imgproc::LINE_8, 
        0
    )?;

    // core::add_weighted(&mat1, 0.2, &mat2, 0.8, 0.0, frame, mat1.depth()?)?;

    Ok(())
}

fn draw_track(frame: &mut Mat, p: &(i32, (na::Point2<f32>, u64, f32)), color: core::Scalar) -> opencv::Result<()> {
    let x = p.1.0.x;
    let y = p.1.0.y;

    // opencv::imgproc::line(
    //     frame, 
    //     core::Point::new(
    //         x as i32, 
    //         y as i32,
    //     ), 
    //     core::Point::new(
    //         (x + p.object.direction.re * 20.0) as i32, 
    //         (y + p.object.direction.im * 20.0) as i32, 
    //     ),
    //     core::Scalar::new(255.0, 255.0, 0.0, 0.0), 
    //     1, 
    //     opencv::imgproc::LINE_8, 
    //     0
    // )?;

    opencv::imgproc::circle(
        frame, 
        core::Point::new(
            x as i32,
            y as i32,
        ), 
        4, 
        color, 
        opencv::imgproc::FILLED, 
        opencv::imgproc::LINE_8, 
        0
    )?;

    opencv::imgproc::put_text(
        frame, 
        &format!("{}", p.0), 
        core::Point::new(x as i32, (y - 10.0) as i32), 
        opencv::imgproc::FONT_HERSHEY_SIMPLEX, 
        0.4, 
        core::Scalar::new(0.0, 0.0, 0.0, 255.0),
        1,
        opencv::imgproc::LINE_AA, 
        false
    )?;

    Ok(())
}

fn main() -> Result<(), grant_tracker::error::Error> {
    use std::io::BufRead;

    let mut args = std::env::args();
    let _ = args.next().unwrap();
    let in_file_name = args.next().expect("expected video file name");
    let track_root = args.next().expect("expected tracks root");
    let track_file_name = track_root + &std::path::Path::new(&in_file_name).file_stem().unwrap().to_string_lossy() + ".track";
    let track_file = std::fs::File::open(track_file_name)?;

    let window = "video capture";
    let points = "points";
    highgui::named_window(window, 1)?;
    // highgui::named_window(points, 1)?;

    let mut cam = videoio::VideoCapture::from_file(&in_file_name, videoio::CAP_ANY)?;  // 0 is the default camera
   
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let mut frames = [core::Mat::default()];

    let mut reader = std::io::BufReader::new(track_file)
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

            Some((id, (na::Point2::new(center_x , center_y), ts, conf)))
        })
        .peekable();

    let mut frame_idx = 0;
    let mut curr_ts = 0;
    loop {
        frame_idx += 1;

        if cam.read(&mut frames[0]).is_err() {
            break;
        }
        
        let (fwidth, fheight) = (frames[0].cols(), frames[0].rows()); 

        if fwidth == 0 || fheight == 0 {
            break;
        }

        let offset = cam.get(videoio::CAP_PROP_POS_MSEC)?;
        let frame = &mut frames[0];
        
        loop {
            let item = if let Some(next) = reader.peek() {
                next
            } else {
                break;
            };

            if item.1.1 != curr_ts {
                curr_ts = item.1.1;
                break;
            }

            let item = reader.next().unwrap();
            draw_track(frame, &item, core::Scalar::new(0.0, 255.0, 0.0, 0.0))?;
        }

        opencv::imgproc::put_text(
            frame, 
            &format!("{}", frame_idx), 
            core::Point::new(10 as i32, 30 as i32), 
            opencv::imgproc::FONT_HERSHEY_SIMPLEX, 
            0.9, 
            core::Scalar::new(255.0, 255.0, 0.0, 255.0),
            1,
            opencv::imgproc::LINE_AA, 
            false
        )?;

        highgui::imshow(window, frame)?;

        let key = highgui::wait_key(10)?;
        if key == 27 {
            break;
        }
    }

    Ok(())
}