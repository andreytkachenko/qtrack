use std::io::Write;
use grant_tracker::meter::yolo_detector::{YoloDetectorConfig, YoloDetector};
use ndarray::prelude::*;
use opencv::{
    dnn,
    core,
    prelude::*,
    videoio,
};
use futures::{pin_mut, stream::{self, TryStreamExt}};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), grant_tracker::error::Error> {
    let mut args = std::env::args();
    let _ = args.next().unwrap();
    let onnx_file_name = args.next().expect("expected onnx file name");
    for in_file_name in args {
        println!("processing: {}", in_file_name);
        
        let out_file_name = std::path::Path::new(&in_file_name).file_stem().unwrap().to_string_lossy().to_string() + ".dets";
        
        if std::path::Path::new(&out_file_name).exists() {
            println!("File {:?} exists - skipping", out_file_name);
            continue;
        }

        let mut out_file = std::fs::File::create(&out_file_name)?;

        let cam = videoio::VideoCapture::from_file(&in_file_name, videoio::CAP_ANY)?;
        let opened = videoio::VideoCapture::is_opened(&cam)?;
        if !opened {
            panic!("Unable to open default camera!");
        }

        let device = onnx_model::get_cuda_if_available(None);
        let mut config = YoloDetectorConfig::new(0.2, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        config.class_map = Some(vec![0, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

        let detector = std::sync::Arc::new(YoloDetector::new(&onnx_file_name, config, device)?);

        let stream = stream::unfold((cam, 0i32), move |(mut cam, idx)| {
            async move {
                let mut mat = core::Mat::default();
                if cam.read(&mut mat).is_err() {
                    return None;
                }
                
                if mat.rows() == 0 || mat.cols() == 0 {
                    return None;
                }

                let offset = cam.get(videoio::CAP_PROP_POS_MSEC).unwrap().round() as u64;

                Some(((mat, offset, idx), (cam, idx + 1)))
            }
        });
        
        const SIZE_W: usize = 800;
        const SIZE_H: usize = 480;
        const CROP: bool = false;

        let stream = stream
                .map(|(frame, offset, frame_idx)| {
                    tokio::task::spawn_blocking(move || {
                        let fsize = frame.size().unwrap();

                        let mut inpt = Array4::zeros([1, 3, SIZE_H, SIZE_W]);
                        let frame_height = fsize.height;
                        let frame_width = fsize.width;

                        let h = SIZE_H;
                        let w = SIZE_W;
                        
                        let blob = dnn::blob_from_image(
                            &frame, 
                            1.0 / 255.0, 
                            core::Size::new(w as _, h as _), 
                            core::Scalar::new(0., 0., 0., 0.), 
                            true, 
                            CROP, 
                            core::CV_32F)
                            .unwrap();
                        
                        let core = blob.try_into_typed::<f32>().unwrap();
                        let view = aview1(core.data_typed().unwrap()).into_shape([3, h, w]).unwrap();
                        inpt.slice_mut(s![0, .., 0..h, 0..w]).assign(&view);

                        (inpt, frame_width, frame_height, offset, frame_idx)
                    })
                })
                .buffered(8)
                .map_ok(|(inpt, frame_width, frame_height, offset, frame_idx)| {
                    let detector = detector.clone();

                    tokio::task::spawn_blocking(move || {
                        let begin = std::time::Instant::now();
                        (
                            detector.detect(inpt.view(), frame_width, frame_height, CROP).unwrap().into_iter().next().unwrap(),
                            offset,
                            frame_idx,
                            begin
                        )
                    })
                })
                .try_buffered(8);

        pin_mut!(stream);

        let mut counter = 0i32;
        let mut count = 0i32;
        let mut begin = std::time::Instant::now();

        while let Some((dets, offset, frame_idx, _)) = stream.try_next().await.unwrap() {
            counter += 1;
            writeln!(out_file, "{}: {}", offset, serde_json::to_string(&dets)?)?;

            let instant = std::time::Instant::now();
            let diff = instant - begin;

            if diff >= std::time::Duration::from_secs(10) {
                let cdiff = (counter - count) as f32;
                let fps = cdiff / diff.as_secs_f32();
                println!("[{}; {}] {:?}", frame_idx, offset, fps);

                count = counter;
                begin = instant;
            }
        }
    }

    Ok(())
}