use grant_tracker::meter::scene::{Participant, Scene};
use grant_common::Detection;
use opencv::{
    core::{self, Mat},
    highgui,
    prelude::*,
    videoio,
};
use nalgebra as na;

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
        (det.x - det.w / 2.0) as i32, 
        (det.y - det.h / 2.0) as i32, 
        det.w as i32, 
        det.h as i32
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

    Ok(())
}

fn main() -> Result<(), grant_tracker::error::Error> {
    use std::io::BufRead;
    let mut args = std::env::args();
    let _ = args.next().unwrap();
    let in_file_name = args.next().expect("expected video file name");
    let dets_root = args.next().expect("expected detections root");
    let dets_file_name = dets_root + &std::path::Path::new(&in_file_name).file_stem().unwrap().to_string_lossy() + ".dets";
    let dets_file = std::fs::File::open(dets_file_name)?;

    let mut cam = videoio::VideoCapture::from_file(&in_file_name, videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let mut frame = core::Mat::default();

    let width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let total = cam.get(videoio::CAP_PROP_FRAME_COUNT)? as i32;

    println!("video {}x{} {} frames", width, height, total);

    let mut writer = VideoWriter::new("out.avi");

    let reader = std::io::BufReader::new(dets_file).lines();
    let mut line_reader = reader.into_iter();
    let mut frame_idx = 0;

    loop {
        frame_idx += 1;
        print!("\rprogress {}/{}...", frame_idx, total);
        if cam.read(&mut frame).is_err() {
            break;
        }
        let (fwidth, fheight) = (frame.cols(), frame.rows()); 
        if fwidth == 0 || fheight == 0 {
            break;
        }

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

        for det in &xx {
            draw_pred(&mut frame, det).unwrap();
        }

        opencv::imgproc::put_text(
            &mut frame, 
            &format!("{}", frame_idx), 
            core::Point::new(10 as i32, 30 as i32), 
            opencv::imgproc::FONT_HERSHEY_SIMPLEX, 
            0.9, 
            core::Scalar::new(255.0, 255.0, 0.0, 255.0),
            1,
            opencv::imgproc::LINE_AA, 
            false
        )?;

        writer.feed(&mut frame);
    }
    println!("\nfinished");

    writer.release();

    Ok(())
}