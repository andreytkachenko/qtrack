use crate::detection::Detection;
use crate::error::Error;

use ndarray::prelude::*;
use onnx_model::*;

const MODEL_DYNAMIC_INPUT_DIMENSION: i64 = -1;

pub struct YoloDetectorConfig {
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub classes: Vec<i32>,
    pub class_map: Option<Vec<i32>>,
}

impl YoloDetectorConfig {
    pub fn new(confidence_threshold: f32, classes: Vec<i32>) -> Self {
        Self {
            confidence_threshold,
            iou_threshold: 0.2,
            classes,
            class_map: None,
        }
    }
}

pub struct YoloDetector {
    model: OnnxInferenceModel,
    config: YoloDetectorConfig,
}

impl YoloDetector {
    pub fn new(
        model_src: &str,
        config: YoloDetectorConfig,
        device: InferenceDevice,
    ) -> Result<Self, Error> {
        let model = OnnxInferenceModel::new(model_src, device)?;

        Ok(Self { model, config })
    }

    pub fn get_model_input_size(&self) -> Option<(u32, u32)> {
        let mut input_dims = self.model.get_input_infos()[0].shape.dims.clone();
        let input_height = input_dims.pop().unwrap();
        let input_width = input_dims.pop().unwrap();
        if input_height == MODEL_DYNAMIC_INPUT_DIMENSION
            && input_width == MODEL_DYNAMIC_INPUT_DIMENSION
        {
            None
        } else {
            Some((input_width as u32, input_height as u32))
        }
    }

    pub fn detect(
        &self,
        frames: ArrayView4<'_, f32>,
        fw: i32,
        fh: i32,
        with_crop: bool,
    ) -> Result<Vec<Vec<Detection>>, Error> {
        let in_shape = frames.shape();
        let (in_w, in_h) = (in_shape[3], in_shape[2]);
        let preditions = self.model.run(&[frames.into_dyn()])?.pop().unwrap();
        let shape = preditions.shape();
        let shape = [shape[0], shape[1], shape[2]];
        let arr = preditions.into_shape(shape).unwrap();
        let bboxes = self.postprocess(arr.view(), in_w, in_h, fw, fh, with_crop)?;

        Ok(bboxes)
    }

    fn postprocess(
        &self,
        view: ArrayView3<'_, f32>,
        in_w: usize,
        in_h: usize,
        frame_width: i32,
        frame_height: i32,
        with_crop: bool,
    ) -> Result<Vec<Vec<Detection>>, Error> {
        let shape = view.shape();
        let nbatches = shape[0];
        let npreds = shape[1];
        let pred_size = shape[2];
        let mut results: Vec<Vec<Detection>> = (0..nbatches).map(|_| vec![]).collect();

        let (ox, oy, ow, oh) = if with_crop {
            let in_a = in_h as f32 / in_w as f32;
            let frame_a = frame_height as f32 / frame_width as f32;

            if in_a > frame_a {
                let w = frame_height as f32 / in_a;
                ((frame_width as f32 - w) / 2.0, 0.0, w, frame_height as f32)
            } else {
                let h = frame_width as f32 * in_a;
                (0.0, (frame_height as f32 - h) / 2.0, frame_width as f32, h)
            }
        } else {
            (0.0, 0.0, frame_width as f32, frame_height as f32)
        };

        // Extract the bounding boxes for which confidence is above the threshold.
        for batch in 0..nbatches {
            let results = &mut results[batch];

            // The bounding boxes grouped by (maximum) class index.
            let mut bboxes: Vec<Vec<Detection>> = (0..80).map(|_| vec![]).collect();

            for index in 0..npreds {
                let x_0 = view.index_axis(Axis(0), batch);
                let x_1 = x_0.index_axis(Axis(0), index);
                let detection = x_1.as_slice().unwrap();

                let (x, y, w, h) = match &detection[0..4] {
                    [center_x, center_y, width, height] => {
                        let center_x = ox + center_x * ow;
                        let center_y = oy + center_y * oh;
                        let width = width * ow as f32;
                        let height = height * oh as f32;

                        (center_x, center_y, width, height)
                    }

                    _ => unreachable!(),
                };

                let classes = &detection[4..pred_size];

                let mut class_index = -1;
                let mut confidence = 0.0;

                for (idx, val) in classes.iter().copied().enumerate() {
                    if val > confidence {
                        class_index = idx as i32;
                        confidence = val;
                    }
                }

                if class_index > -1 && confidence > self.config.confidence_threshold {
                    if !self.config.classes.contains(&class_index) {
                        continue;
                    }

                    if w * h > ((frame_width / 2) * (frame_height / 2)) as f32 {
                        continue;
                    }

                    let mapped_class = match &self.config.class_map {
                        Some(map) => map
                            .get(class_index as usize)
                            .copied()
                            .unwrap_or(class_index),
                        None => class_index,
                    };

                    bboxes[mapped_class as usize].push(Detection {
                        x,
                        y,
                        w,
                        h,
                        confidence,
                        class: class_index as _,
                    });
                }
            }

            for mut dets in bboxes.into_iter() {
                if dets.is_empty() {
                    continue;
                }

                if dets.len() == 1 {
                    results.append(&mut dets);
                    continue;
                }

                let indices = self.non_maximum_supression(&mut dets)?;

                results.extend(dets.drain(..).enumerate().filter_map(|(idx, item)| {
                    if indices.contains(&(idx as i32)) {
                        Some(item)
                    } else {
                        None
                    }
                }));

                // for (det, idx) in dets.into_iter().zip(indices) {
                //     if idx > -1 {
                //         results.push(det);
                //     }
                // }
            }
        }

        Ok(results)
    }

    // fn non_maximum_supression(&self, dets: &mut [Detection]) -> Result<Vec<i32>, Error> {
    //     let mut rects = core::Vector::new();
    //     let mut scores = core::Vector::new();

    //     for det in dets {
    //         rects.push(core::Rect2d::new(
    //             det.xmin as f64,
    //             det.ymin as f64,
    //             (det.xmax - det.xmin) as f64,
    //             (det.ymax - det.ymin) as f64
    //         ));
    //         scores.push(det.confidence);
    //     }

    //     let mut indices = core::Vector::<i32>::new();
    //     dnn::nms_boxes_f64(
    //         &rects,
    //         &scores,
    //         self.config.confidence_threshold,
    //         self.config.iou_threshold,
    //         &mut indices,
    //         1.0,
    //         0
    //     )?;

    //     Ok(indices.to_vec())
    // }

    fn non_maximum_supression(&self, dets: &mut [Detection]) -> Result<Vec<i32>, Error> {
        dets.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut retain: Vec<_> = (0..dets.len() as i32).collect();
        for idx in 0..dets.len() - 1 {
            if retain[idx] != -1 {
                for r in retain[idx + 1..].iter_mut() {
                    if *r != -1 {
                        let iou = dets[idx].iou(&dets[*r as usize]);
                        if iou > self.config.iou_threshold {
                            *r = -1;
                        }
                    }
                }
            }
        }

        retain.retain(|&x| x > -1);
        Ok(retain)
    }
}
