use anyhow::Error;
use grant_object_detector::{Detection, YoloDetector, YoloDetectorConfig};

use tracker::deep_sort::{sort, DeepSortConfig, DeepSort, Track};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use ndarray::prelude::*;
use opencv::{
    dnn,
    core::{self, Mat, Scalar, Vector},
    highgui,
    prelude::*,
    videoio,
};

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

fn draw_pred(frame: &mut Mat, det: &ObjectDetection) -> opencv::Result<()> {
    let rect = core::Rect::new(
        det.bbox[0] as i32,
        det.bbox[1] as i32,
        (det.bbox[2] - det.bbox[0]) as i32,
        (det.bbox[3] - det.bbox[1]) as i32
    );

    let color = match det.class {
        0 => core::Scalar::new(0.0, 0.0, 255.0, 0.0), // red
        1 => core::Scalar::new(0.0, 255.0, 0.0, 0.0), // green
        2 => core::Scalar::new(255.0, 0.0, 0.0, 0.0), // blue
        3 => core::Scalar::new(0.0, 255.0, 255.0, 0.0), // yellow
        5 => core::Scalar::new(255.0, 0.0, 255.0, 0.0), // magenta
        7 => core::Scalar::new(255.0, 255.0, 0.0, 0.0), // cyan
        _ => core::Scalar::new(0.0, 0.0, 0.0, 0.0), // black
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
    Ok(())
}


fn draw_track(frame: &mut Mat, track_id: i32, track: &ObjectTrack, info: &ObjectInfo, highlighted: bool) -> opencv::Result<()> {
    let color = if highlighted {
        core::Scalar::new(0.0, 0.0, 255.0, 128.0)
    } else {
        core::Scalar::new(0.0, 128.0, 255.0, 128.0)
    };

    let rect = opencv::core::Rect::new(
        track.bbox[0] as i32,
        track.bbox[1] as i32,
        (track.bbox[2] - track.bbox[0]) as i32,
        (track.bbox[3] - track.bbox[1]) as i32
    );

    //  Draw a bounding box.
    opencv::imgproc::rectangle(
        frame,
        rect,
        color,
        1,
        opencv::imgproc::LINE_8,
        0
    )?;

    let label = format!("{}", track_id);
    let kind = format!("{}", NAMES[info.class as usize]);

    let mut base_line = 0;
    let label_size = opencv::imgproc::get_text_size(&label, opencv::imgproc::FONT_HERSHEY_SIMPLEX, 0.4, 1, &mut base_line)?;
    let kind_size = opencv::imgproc::get_text_size(&kind, opencv::imgproc::FONT_HERSHEY_SIMPLEX, 0.4, 1, &mut base_line)?;

    let label_rect = core::Rect::new(
        rect.x,
        rect.y,
        rect.width,
        label_size.height + 4
    );

    let kind_rect = core::Rect::new(
        rect.x + rect.width - kind_size.width,
        rect.y,
        kind_size.width,
        kind_size.height + 4
    );

    opencv::imgproc::rectangle(frame, label_rect, color, opencv::imgproc::FILLED, opencv::imgproc::LINE_8, 0)?;
    // opencv::imgproc::rectangle(frame, kind_rect, core::Scalar::new(0.0, 128.0, 255.0, 255.0), opencv::imgproc::FILLED, opencv::imgproc::LINE_8, 0)?;

    opencv::imgproc::put_text(
        frame,
        &label,
        core::Point::new(rect.x, rect.y + label_size.height),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.4,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        1,
        opencv::imgproc::LINE_AA,
        false
    )?;

    opencv::imgproc::put_text(
        frame,
        &kind,
        core::Point::new(kind_rect.x, kind_rect.y + kind_size.height),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.4,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        1,
        opencv::imgproc::LINE_AA,
        false
    )?;

    Ok(())
}

pub enum MarkerResult {
    Absent,
    Weak,
    Strong,
}

pub struct Incedent {
    id: i64,
    is_confirmed: bool,
    objects: Vec<i32>,

    door_marker: MarkerResult,
    damage_marker: MarkerResult,
    people_marker: MarkerResult,
    speed_marker: MarkerResult,
    alarm_marker: MarkerResult,
    time_marker: MarkerResult,
}

#[derive(Serialize, Deserialize)]
pub struct ObjectInfo {
    class: i32,
}

#[derive(Serialize, Deserialize)]
pub struct ObjectTrack {
    direction: [f32; 2],
    bbox: [f32; 4],
}

#[derive(Serialize, Deserialize)]
pub struct ObjectDetection {
    confidence: f32,
    class: i32,
    bbox: [f32; 4],
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum MarkerValue {
    NotDetected,
    Weak,
    Strong,
    Extra,
}

impl Default for MarkerValue {
    fn default() -> Self {
        MarkerValue::NotDetected
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct IncidentInfo {
    id: (i32, i32),
    doors: MarkerValue,
    alarm: MarkerValue,
    damage: MarkerValue,
    people: MarkerValue,
    speed: MarkerValue,
    time: MarkerValue,
}

impl IncidentInfo {
    pub fn new(a: i32, b: i32) -> Self {
        Self {
            id: (a, b),
            doors: Default::default(),
            alarm: Default::default(),
            damage: Default::default(),
            people: Default::default(),
            speed: Default::default(),
            time: Default::default(),
        }
    }
}


#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum IncidentUpdateKind {
    Doors,
    Alarm,
    Damage,
    People,
    Speed,
    Time,
}

#[derive(Serialize, Deserialize)]
pub struct IncidentUpdate {
    id: (i32, i32),
    kind: IncidentUpdateKind,
    value: MarkerValue,
}

impl IncidentUpdate {
    pub fn new(id: (i32, i32), kind: IncidentUpdateKind, value: MarkerValue) -> Self {
        Self {
            id,
            value,
            kind,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct IncidentConfirm {
    id: (i32, i32),
}

impl IncidentConfirm {
    pub fn new(a: i32, b: i32) -> Self {
        Self {
            id: (a, b),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct IncidentDelete {
    id: (i32, i32),
}

impl IncidentDelete {
    pub fn new(a: i32, b: i32) -> Self {
        Self {
            id: (a, b),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum ObjectEvent {
    Intersection(i32, i32),
    IncidentCreate(IncidentInfo),
    IncidentUpdate(IncidentUpdate),
    IncidentConfirm(IncidentConfirm),
    IncidentDelete(IncidentDelete),
}

#[derive(Default, Serialize, Deserialize)]
pub struct Frame {
    tracks: HashMap<i32, ObjectTrack>,
    detections: Vec<ObjectDetection>,
    events: Vec<ObjectEvent>,
}

#[derive(Serialize, Deserialize)]
pub struct Map {
    frames: HashMap<i64, Frame>,
    objects: HashMap<i32, ObjectInfo>,
    classes: HashMap<i32, String>,
}

impl Map {
    pub fn new() -> Self {
        Self {
            frames: HashMap::new(),
            objects: HashMap::new(),
            classes: HashMap::new(),
        }
    }

    pub fn object_info(&self, track_id: i32) -> Option<&ObjectInfo> {
        self.objects.get(&track_id)
    }

    pub fn events(&self, frame_id: i64) -> &[ObjectEvent] {
        if let Some(frame) = self.frames.get(&frame_id) {
            frame.events.as_slice()
        } else {
            &[]
        }
    }

    pub fn detections(&self, frame_id: i64) -> &[ObjectDetection] {
        if let Some(frame) = self.frames.get(&frame_id) {
            frame.detections.as_slice()
        } else {
            &[]
        }
    }

    pub fn tracks(&self, frame_id: i64) -> Vec<(i32, &ObjectTrack)> {
        if let Some(frame) = self.frames.get(&frame_id) {
            frame.tracks.iter().map(|(&k, v)| (k, v)).collect()
        } else {
            vec![]
        }
    }

    pub fn add_event(&mut self, frame_id: i64, event: ObjectEvent) {
        let frame = self.frames.entry(frame_id).or_insert_with(Default::default);

        frame.events.push(event);
    }

    pub fn add_detection(&mut self, frame_id: i64, det: &Detection) {
        let frame = self.frames.entry(frame_id).or_insert_with(Default::default);
        self.classes
            .entry(det.class)
            .or_insert_with(||NAMES[det.class as usize].to_string());

        frame.detections.push(ObjectDetection {
            confidence: det.confidence,
            class: det.class,
            bbox: [det.x, det.y, det.w, det.h],
        });
    }

    pub fn add_tarck(&mut self, frame_id: i64, track: &sort::Track) {
        let frame = self.frames.entry(frame_id).or_insert_with(Default::default);

        let object = self.objects
            .entry(track.track_id)
            .or_insert_with(||ObjectInfo { class: track.class() });

        object.class = track.class();

        self.classes
            .entry(object.class)
            .or_insert_with(||NAMES[object.class as usize].to_string());

        frame.tracks.insert(track.track_id, ObjectTrack {
            direction: [0.0f32, 0.0f32],
            bbox: track.bbox().as_ltrb().into(),
        });
    }
}

fn main() -> Result<(), anyhow::Error> {
    let mut writer = VideoWriter::new("out.avi");
    let map_file = std::fs::File::open("out.json")?;
    let map: Map = serde_json::from_reader(map_file)?;

    let mut frame_idx = 0i64;

    let mut cam = videoio::VideoCapture::from_file("/home/andrey/test_video/1-3.mp4", videoio::CAP_ANY)?;  // 0 is the default camera
    // cam.set(videoio::CAP_PROP_POS_FRAMES, 150.0);

    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let mut frames = [core::Mat::default()];
    let mut context: Vec<(bool, IncidentInfo)> = Vec::new();
    let mut fq = [0f64; 64];
    let mut counter = 0;

    loop {
        frame_idx += 1;
        if cam.read(&mut frames[0]).is_err() {
            break;
        }

        if frames[0].rows() == 0 || frames[0].cols() == 0 {
            break;
        }

        println!("frame #{}:", frame_idx);

        let frame = &mut frames[0];

        let dets = map.detections(frame_idx);
        for d in dets {
            if d.class == 0 {
                let _ = draw_pred(frame, d);
            }
        }

        let tracks = map.tracks(frame_idx);
        for &(track_id, track) in &tracks {
            let info = map.object_info(track_id).unwrap();
            let highlighted = {
                let mut res = false;

                for (_, c) in &context {
                    if c.id.0 == track_id || c.id.1 == track_id {
                        res = true;
                        break;
                    }
                }

                res
            };

            let _ = draw_track(frame, track_id, track, info, highlighted);
        }


        if let Some(&(_, track)) = tracks.iter().find(|&&(track_id, _)| track_id == 18) {
            let [left, top, width, height] = track.bbox;

            let left = (left as i32).max(0).min(frame.cols());
            let top = (top as i32).max(0).min(frame.rows());

            let mut width = (width as i32).max(0);
            let mut height = (height as i32).max(0);

            if left + width > frame.cols() {
                width = frame.cols() - left;
            }

            if top + height > frame.rows() {
                height = frame.rows() - top;
            }

            let rect = core::Rect::new(
                left,
                top,
                width,
                height,
            );

            let roi = Mat::roi(&frame, rect).unwrap().clone();
            let tmp = roi.try_into_typed::<core::Vec3b>().unwrap();
            let x = tmp.data_typed().unwrap();
            let mut fqt = [0f64; 64];

            let mut avg = 0f64;
            for i in x {
                avg += (i.0[0] + i.0[1] + i.0[2]) as f64;
            }

            avg /= (x.len() * 3) as f64;

            for j in 0..64 {
                if counter > 0 && counter % (j + 1) == 0 {
                    fqt[j as usize] += avg;
                }
            }

            for j in 0..64 {
                fqt[j as usize] /= (64 - j) as f64 / 64.0;
            }

            if counter % 300 == 0 {
                println!("{:.3?}", fq);
                for j in 0..64 {
                    fq[j as usize] = fqt[j as usize];
                }
            } else {
                for j in 0..64 {
                    fq[j as usize] += fqt[j as usize];
                }
            }

            counter += 1;
        }

        let events = map.events(frame_idx);
        for event in events {
            match event {
                ObjectEvent::IncidentCreate(info) => context.push((false, info.clone())),
                ObjectEvent::IncidentUpdate(info) => {
                    for i in &mut context {
                        if i.1.id == info.id {
                            match info.kind {
                                IncidentUpdateKind::Doors => i.1.doors = info.value,
                                IncidentUpdateKind::Alarm => i.1.alarm = info.value,
                                IncidentUpdateKind::Damage => i.1.damage = info.value,
                                IncidentUpdateKind::People => i.1.people = info.value,
                                IncidentUpdateKind::Speed => i.1.speed = info.value,
                                IncidentUpdateKind::Time => i.1.time = info.value,
                            }
                        }
                    }
                },
                ObjectEvent::IncidentConfirm(info) => {
                    for i in &mut context {
                        if i.1.id == info.id {
                            i.0 = true;
                        }
                    }
                },
                _ => (),
            }
        }

        for (idx, ctx) in context.iter().enumerate() {
            let color = core::Scalar::new(128.0, 128.0, 255.0, 255.0);

            let base_rect = core::Rect::new(
                940,
                40 + (idx * 200) as i32,
                300,
                160,
            );
            let mut smaller_rect = base_rect.clone();
            smaller_rect.height = 20;

            opencv::imgproc::rectangle(frame, smaller_rect, color, opencv::imgproc::FILLED, opencv::imgproc::LINE_8, 0)?;

            opencv::imgproc::rectangle(
                frame,
                base_rect,
                color,
                1,
                opencv::imgproc::LINE_8,
                0
            )?;

            opencv::imgproc::put_text(
                frame,
                &format!("{}incident_{}-{}", if ctx.0 {"[confirmed] "} else {""}, ctx.1.id.0, ctx.1.id.1),
                core::Point::new(950, 54 + (idx * 200) as i32),
                opencv::imgproc::FONT_HERSHEY_SIMPLEX,
                0.4,
                core::Scalar::new(0.0, 0.0, 0.0, 255.0),
                1,
                opencv::imgproc::LINE_AA,
                false
            ).unwrap();

            draw_cb(frame, 948, 78 + (idx * 200) as i32, ctx.1.doors, "Opened Doors", color_by_marker(ctx.1.doors));
            draw_cb(frame, 948, 98 + (idx * 200) as i32, ctx.1.speed, "Car Direction Change", color_by_marker(ctx.1.speed));
            draw_cb(frame, 948, 118 + (idx * 200) as i32, ctx.1.alarm, "Car Alarm", color_by_marker(ctx.1.alarm));
            draw_cb(frame, 948, 138 + (idx * 200) as i32, ctx.1.damage, "Car Damage", color_by_marker(ctx.1.damage));
            draw_cb(frame, 948, 158 + (idx * 200) as i32, ctx.1.people, "People Around", color_by_marker(ctx.1.people));
            draw_cb(frame, 948, 178 + (idx * 200) as i32, ctx.1.time, "Time Past", color_by_marker(ctx.1.time));
        }

        writer.feed(frame);
    }

    writer.release();

    Ok(())
}

fn color_by_marker(val: MarkerValue) -> core::Scalar {
    match val {
        MarkerValue::NotDetected => core::Scalar::new(0.0, 0.0, 255.0, 255.0),
        MarkerValue::Weak => core::Scalar::new(0.0, 255.0, 255.0, 255.0),
        MarkerValue::Strong => core::Scalar::new(0.0, 255.0, 0.0, 255.0),
        MarkerValue::Extra => core::Scalar::new(255.0, 255.0, 0.0, 255.0)
    }
}


fn draw_cb(frame: &mut core::Mat, x: i32, y: i32, val: MarkerValue, title: &str, color: core::Scalar) {
    let p = match val {
        MarkerValue::NotDetected => "  No",
        MarkerValue::Weak => "Weak",
        MarkerValue::Strong => "Good",
        MarkerValue::Extra => "Exel"
    };

    opencv::imgproc::put_text(
        frame,
        &format!("{}", p),
        core::Point::new(x, y),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        opencv::imgproc::LINE_8,
        false
    ).unwrap();

    opencv::imgproc::put_text(
        frame,
        &format!("{}", title),
        core::Point::new(x + 60, y),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        core::Scalar::new(255.0, 255.0, 0.0, 255.0),
        1,
        opencv::imgproc::LINE_8,
        false
    ).unwrap();
}
