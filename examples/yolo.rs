use grant_common::Detection;
use grant_tracker::meter::yolo_detector::{YoloDetectorConfig, YoloDetector};
use ndarray::prelude::*;
use opencv::{
    dnn,
    core,
    highgui,
    prelude::*,
    videoio,
};
use futures::{pin_mut, stream::{self, TryStreamExt}};
use futures::StreamExt;

pub const NAMES: [&'static str; 8] = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "bus",
    "truck",
    "unk",
    "unk",
];


fn draw_pred(frame: &mut Mat, det: &Detection) -> opencv::Result<()> {
    let w2 = det.w / 2.0;
    let h2 = det.h / 2.0;
    let rect = core::Rect::new(
        (det.x - w2) as i32, 
        (det.y - h2) as i32, 
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

    opencv::imgproc::put_text(
        frame, 
        &format!("{}", det.class), 
        core::Point::new(det.x as i32, det.y as i32), 
        opencv::imgproc::FONT_HERSHEY_SIMPLEX, 
        0.9, 
        core::Scalar::new(255.0, 255.0, 0.0, 255.0),
        1,
        opencv::imgproc::LINE_AA, 
        false
    )?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), grant_tracker::error::Error> {
    let mut args = std::env::args();
    let _ = args.next().unwrap();
    let onnx_file_name = args.next().expect("expected onnx file name");
    let in_file_name = args.next().expect("expected video file name");

    let cam = videoio::VideoCapture::from_file(&in_file_name, videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let window = "video capture";
    highgui::named_window(window, 1)?;

    let device = onnx_model::get_cuda_if_available(None);
    let mut config = YoloDetectorConfig::new(0.1, vec![0, 1, 2, 3, 4, 5]);
    config.class_map = Some(vec![0, 1, 1, 1, 1, 1]);

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

    // let (model_w, model_h) = detector.get_model_input_size().unwrap();
    
    const SIZE_W: usize = 800;
    const SIZE_H: usize = 480;
    const CROP: bool = false;

    let stream = stream
            .map(|(frame, offset, frame_idx)| {
                let detector = detector.clone();

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

                    // blob.assign_to(&mut frame, core::CV_8U).unwrap();
                    let core = blob.try_into_typed::<f32>().unwrap();
                    let view = aview1(core.data_typed().unwrap()).into_shape([3, h, w]).unwrap();
                    inpt.slice_mut(s![0, .., 0..h, 0..w]).assign(&view);

                    (
                        frame,
                        detector.detect(inpt.view(), frame_width, frame_height, CROP).unwrap().into_iter().next().unwrap(),
                        offset,
                        frame_idx
                    )
                })
            })
            .buffered(8);

    pin_mut!(stream);

    while let Some((mut frame, dets, _, frame_idx)) = stream.try_next().await.unwrap() {
        for det in &dets {
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

        highgui::imshow(window, &frame).unwrap();

        let key = highgui::wait_key(10)?;
        if key == 27 {
            break;
        }
    }

    Ok(())
}