use qtrack::scene::{Participant, Scene};
use qtrack::Detection;
use opencv::{
    core::{self, Mat},
    highgui,
    prelude::*,
    videoio,
};
use nalgebra as na;
use clap::{Clap, AppSettings};

pub struct VideoWriter {
    writer: Option<opencv::videoio::VideoWriter>,
    size: Option<(i32, i32)>,
    out_file: String
}

impl VideoWriter {
    pub fn new<S: ToString>(out_file: S) -> Self {
        Self {
            writer: None,
            size: None,
            out_file: out_file.to_string()
        }
    }

    pub fn release(&mut self) {
        if let Some(mut w) = self.writer.take() {
            w.release().unwrap();
        }
    }

    fn reinit(&mut self, size: (i32, i32)) {
        println!("reinit {:?}", size);
        self.release();

        self.size = Some(size);
        self.writer = Some(opencv::videoio::VideoWriter::new(
            &self.out_file,
            opencv::videoio::VideoWriter::fourcc(b'X' as _, b'V' as _, b'I' as _, b'D' as _).unwrap(),
            24.0,
            opencv::core::Size::new(size.0, size.1),
            true,
        ).unwrap());
    }

    pub fn feed(&mut self, m: &mut opencv::core::Mat)  {
        let size = (m.cols(), m.rows());

        if self.writer.is_none() || self.size != Some(size) {
            self.reinit(size);
        }

        self.writer.as_mut().unwrap().write(m).unwrap();
    }
}

pub const NAMES: [&'static str; 8] = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
];

fn draw_pred(frame: &mut Mat, det: &Detection) -> opencv::Result<()> {
    let rect = core::Rect::new(
        (det.x - 2.0) as i32,
        (det.y - 2.0) as i32,
        4 as i32,
        4 as i32
    );

    let color = match det.class {
        0 => core::Scalar::new(200.0, 0.0, 0.0, 128.0),
        1 => core::Scalar::new(255.0, 255.0, 0.0, 128.0),
        2 => core::Scalar::new(0.0, 200.0, 200.0, 128.0),
        3 => core::Scalar::new(200.0, 0.0, 200.0, 128.0),
        4 => core::Scalar::new(100.0, 100.0, 0.0, 128.0),
        5 => core::Scalar::new(200.0, 200.0, 200.0, 128.0),
        6 => core::Scalar::new(0.0, 0.0, 0.0, 128.0),
        7 => core::Scalar::new(0.0, 0.0, 0.0, 128.0),
        8 => core::Scalar::new(0.0, 0.0, 0.0, 128.0),
        _ => core::Scalar::new(0.0, 0.0, 0.0, 128.0),
    };

    //  Draw a bounding box.
    opencv::imgproc::rectangle(
        frame,
        rect,
        color,
        1,
        opencv::imgproc::LINE_8,
        0
    )?;

    // core::add_weighted(&mat1, 0.2, &mat2, 0.8, 0.0, frame, mat1.depth()?)?;

    Ok(())
}

fn draw_track(frame: &mut Mat, p: &Participant, color: core::Scalar, ts_sec: f32, detail: bool, rectangles: bool, track: bool) -> opencv::Result<()> {
    let x = p.object.pos.x;
    let y = p.object.pos.y;
    let pred_dist = p.object.predict_distance(0, ts_sec);

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
        &format!("{}", p.id),
        core::Point::new(x as i32, (y - 10.0) as i32),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.4,
        core::Scalar::new(0.0, 0.0, 0.0, 255.0),
        1,
        opencv::imgproc::LINE_AA,
        false
    )?;

    // let value = grant_tracker::meter::math::linearity(p.object.vel_hist.iter_points());
    let speed = (p.object.vel).round();

    opencv::imgproc::put_text(
        frame,
        // &format!("{:0.2}", value),
        &format!("{}", speed),
        core::Point::new(x as i32 - 5, y as i32 + 10),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.3,
        core::Scalar::new(0.0, 128.0, 255.0, 255.0),
        1,
        opencv::imgproc::LINE_AA,
        false
    )?;

    if rectangles {
        let det = p.last_detection();
        // //  Draw a bounding box.
        opencv::imgproc::rectangle(
            frame,
            core::Rect::new(
                (det.x - det.w * 0.5) as i32,
                (det.y - det.h * 0.5) as i32,
                det.w as i32,
                det.h as i32
            ),
            core::Scalar::new(255.0, 255.0, 255.0, 128.0),
            1,
            opencv::imgproc::LINE_8,
            0
        )?;
    }

    if track {
        for (idx, &(_, pt, _)) in p.object.history.iter().enumerate() {
            opencv::imgproc::circle(
                frame,
                core::Point::new(
                    pt.x as i32,
                    pt.y as i32,
                ),
                1,
                core::Scalar::new(255.0, 128.0, 255.0, 0.0),
                opencv::imgproc::FILLED,
                opencv::imgproc::LINE_8,
                0
            )?;
        }
    }

    if detail {
        if p.object.predictor.has_quadratic {
            let mut prev: Option<na::Complex<f32>> = None;
            for x in -30 .. (pred_dist as i32 + 120) {
                let x = x as f32;
                let y = p.object.predictor.curvature * x * x;
                let pt = na::Complex::new(x, y) * p.object.predictor.direction + p.object.predictor.extremum;

                if let Some(prev) = prev {
                    opencv::imgproc::line(
                        frame,
                        core::Point::new(
                            prev.re as i32,
                            prev.im as i32,
                        ),
                        core::Point::new(
                            pt.re as i32,
                            pt.im as i32,
                        ),
                        core::Scalar::new(255.0, 255.0, 0.0, 0.0),
                        1,
                        opencv::imgproc::LINE_8,
                        0
                    )?;
                }

                prev = Some(pt);
            }

            opencv::imgproc::circle(
                frame,
                core::Point::new(
                    p.object.predictor.extremum.re as i32,
                    p.object.predictor.extremum.im as i32,
                ),
                2,
                core::Scalar::new(0.0, 255.0, 255.0, 128.0),
                opencv::imgproc::FILLED,
                opencv::imgproc::LINE_8,
                0
            )?;
        }
    }

    if p.object.predictor.has_linear {
        let line = p.object.predictor.direction;

        // opencv::imgproc::line(
        //     frame,
        //     core::Point::new(
        //         (p.object.predictor.mean.re - line.re * 50.0) as i32,
        //         (p.object.predictor.mean.im - line.im * 50.0) as i32,
        //     ),
        //     core::Point::new(
        //         (p.object.predictor.mean.re + line.re * 50.0) as i32,
        //         (p.object.predictor.mean.im + line.im * 50.0) as i32,
        //     ),
        //     core::Scalar::new(128.0, 128.0, 0.0, 0.0),
        //     1,
        //     opencv::imgproc::LINE_8,
        //     0
        // )?;

        opencv::imgproc::line(
            frame,
            core::Point::new(
                x as i32,
                y as i32,
            ),
            core::Point::new(
                (x + line.re * 30.0) as i32,
                (y + line.im * 30.0) as i32,
            ),
            core::Scalar::new(128.0, 128.0, 0.0, 0.0),
            1,
            opencv::imgproc::LINE_8,
            0
        )?;

        opencv::imgproc::circle(
            frame,
            core::Point::new(
                p.object.predictor.mean.re as i32,
                p.object.predictor.mean.im as i32,
            ),
            2,
            core::Scalar::new(255.0, 0.0, 255.0, 128.0),
            opencv::imgproc::FILLED,
            opencv::imgproc::LINE_8,
            0
        )?;
    }

    let pred = p.object.predict(0, ts_sec);
    opencv::imgproc::circle(
        frame,
        core::Point::new(
            pred.x as i32,
            pred.y as i32,
        ),
        2,
        core::Scalar::new(0.0, 0.0, 255.0, 128.0),
        opencv::imgproc::FILLED,
        opencv::imgproc::LINE_8,
        0
    )?;

    Ok(())
}

fn get_bounding_for_video(name: &str, width: f32, height: f32) -> Vec<na::Point2<f32>> {
    match name {
        _ => {
            let padding = width * 0.01;

            vec![
                nalgebra::Point2::new(padding, padding),
                nalgebra::Point2::new(width - padding - padding, padding),
                nalgebra::Point2::new(width - padding - padding, height - padding - padding),
                nalgebra::Point2::new(padding, height - padding - padding),
            ]
        }
    }
}

#[derive(Debug, Clap)]
#[clap(version = "1.0", author = "Andrey T. <andrey@aidev.ru>")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
    #[clap(short, long)]
    skip: Option<u32>,

    #[clap(short, long)]
    root: String,

    #[clap(short, long)]
    input: String,
}

fn main() -> Result<(), grant_tracker::error::Error> {
    use std::io::BufRead;

    let opts: Opts = Opts::parse();

    let file_stem = std::path::Path::new(&opts.input).file_stem().unwrap().to_string_lossy();
    let dets_file_name = opts.root + &file_stem + ".dets";
    let dets_file = std::fs::File::open(dets_file_name)?;

    let window = "video capture";
    highgui::named_window(window, 1)?;

    let mut cam = videoio::VideoCapture::from_file(&opts.input, videoio::CAP_ANY)?;  // 0 is the default camera

    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let width  = cam.get(videoio::CAP_PROP_FRAME_WIDTH)? as f32;
    let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)? as f32;

    let mut scene = Scene::new(get_bounding_for_video(&file_stem, width, height));

    // let mut points_frame = core::Mat::new_rows_cols_with_default(1024, 1024,  core::CV_8UC3, core::Scalar::new(0.0, 0.0, 0.0, 0.0)).unwrap();
    let mut frames = [core::Mat::default()];
    let reader = std::io::BufReader::new(dets_file).lines();
    let mut line_reader = reader.into_iter();
    let mut frame_idx = 0usize;
    let mut paused = true;
    let mut skip = false;
    let mut rectangles = false;
    let mut details = true;
    let mut tracks = false;
    let mut ghosts = false;
    let mut frame_skip: usize = 1;
    let mut forwarding = opts.skip.unwrap_or(0);

    loop {
        match highgui::wait_key(1)? {
            /* esc */ 27  => break,
            /* spc */ 32  => paused = !paused,
            /*  n  */ 110 => skip = true,
            /*  c  */ 99  => details = !details,
            /*  t  */ 116 => tracks = !tracks,
            /*  b  */ 98  => rectangles = !rectangles,
            /*  z  */ 122 => ghosts = !ghosts,
            /* rgt */ 83  => forwarding = 100,
            /* 1-9 */ code @ 49 ..= 57 => frame_skip = code as usize - 48,
            -1 => (),
            key => println!("key {}", key),
        }

        if paused && frame_idx != 0 {
            if !skip {
                continue;
            } else {
                skip = false;
            }
        }

        frame_idx += 1;

        if cam.read(&mut frames[0]).is_err() {
            break;
        }

        let (fwidth, fheight) = (frames[0].cols(), frames[0].rows());

        if fwidth == 0 || fheight == 0 {
            break;
        }

        let ts_sec =(cam.get(videoio::CAP_PROP_POS_MSEC)? / 1000.0) as f32;
        let frame = &mut frames[0];

        // opencv::imgproc::rectangle(
        //     &mut points_frame,
        //     core::Rect::new(
        //         0,
        //         0,
        //         1024,
        //         1024
        //     ),
        //     core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        //     opencv::imgproc::FILLED,
        //     opencv::imgproc::LINE_8,
        //     0
        // )?;

        let xx: Vec<Detection> = match line_reader.next() {
            Some(x) => {
                let x = x.unwrap();

                if let Some(idx) = x.find(':') {
                    let (_, vector) = x.split_at(idx + 1);
                    serde_json::from_str(vector).unwrap()
                } else {
                    Vec::new()
                }
            },
            None => break,
        };

        if frame_idx % frame_skip == 0 {
            // println!("frame #{}", frame_idx);
            let mapping = scene.map_detections(ts_sec, &xx);
            scene.update(mapping);
        }

        if forwarding > 0 {
            forwarding -= 1;
            continue;
        }

        for t in &scene.tracks {
            if t.id > 0 {
                draw_track(frame, t, core::Scalar::new(0.0, 255.0, 0.0, 0.0), ts_sec, details, rectangles, tracks)?;
            } else if ghosts {
                draw_track(frame, t, core::Scalar::new(128.0, 128.0, 128.0, 0.0), ts_sec, false, false, false)?;
            }
        }

        /*
        for t in &scene.peds {
            if t.id > 0 {
                draw_track(frame, t, core::Scalar::new(0.0, 0.0, 255.0, 0.0), ts_sec, false, false, false)?;
            }
        }
        */

        for det in &xx {
            draw_pred(frame, det).unwrap();
        }

        opencv::imgproc::put_text(
            frame,
            &format!("#{}", frame_idx),
            core::Point::new(10 as i32, 30 as i32),
            opencv::imgproc::FONT_HERSHEY_SIMPLEX,
            0.9,
            core::Scalar::new(255.0, 255.0, 0.0, 255.0),
            1,
            opencv::imgproc::LINE_AA,
            false
        )?;

        highgui::imshow(window, frame)?;
        // highgui::imshow(points, &points_frame)?;
    }

    Ok(())
}
