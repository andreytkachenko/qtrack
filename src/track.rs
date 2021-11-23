use crate::bbox::{BBox, Xyah};

#[derive(Debug, Clone)]
pub struct Track {
    pub track_id: i32,
    pub time_since_update: i32,
    pub class: i32,
    pub confidence: f32,
    pub iou_slip: f32,
    pub bbox: BBox<Xyah>,

    // in px
    pub velocity: Option<f32>,

    // (x,y)
    pub direction: Option<(f32, f32)>,

    // a-coeff from parabolic curve fitting for this track's trajectory
    pub curvature: Option<f32>,
}
