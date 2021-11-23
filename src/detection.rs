use serde_derive::{Deserialize, Serialize};

use crate::bbox::{BBox, Xywh};

/// Contains (x,y) of the center and (width,height) of bbox
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    #[serde(rename = "p")]
    pub confidence: f32,
    #[serde(rename = "c")]
    pub class: i32,
}

impl Detection {
    pub fn iou(&self, other: &Detection) -> f32 {
        let b1_area = (self.w + 1.) * (self.h + 1.);
        let (xmin, xmax, ymin, ymax) = (self.xmin(), self.xmax(), self.ymin(), self.ymax());

        let b2_area = (other.w + 1.) * (other.h + 1.);

        let i_xmin = xmin.max(other.xmin());
        let i_xmax = xmax.min(other.xmax());
        let i_ymin = ymin.max(other.ymin());
        let i_ymax = ymax.min(other.ymax());
        let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);

        (i_area) / (b1_area + b2_area - i_area)
    }

    #[inline(always)]
    pub fn bbox(&self) -> BBox<Xywh> {
        BBox::xywh(self.x, self.y, self.w, self.h)
    }

    #[inline(always)]
    pub fn xmax(&self) -> f32 {
        self.x + self.w / 2.
    }

    #[inline(always)]
    pub fn ymax(&self) -> f32 {
        self.y + self.h / 2.
    }

    #[inline(always)]
    pub fn xmin(&self) -> f32 {
        self.x - self.w / 2.
    }

    #[inline(always)]
    pub fn ymin(&self) -> f32 {
        self.y - self.h / 2.
    }
}
