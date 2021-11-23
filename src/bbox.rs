use serde::{Deserialize, Serialize};
use serde_derive::{Deserialize, Serialize};
use std::marker::PhantomData;

pub trait BBoxFormat: std::fmt::Debug {}

/// Left-top-width-height format, contains left top corner and width-height
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Ltwh;
impl BBoxFormat for Ltwh {}

/// X-y-aspect_ratio-height format, contains coordinates of the center of bbox and aspect_ratio-height
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Xyah;
impl BBoxFormat for Xyah {}

/// Left-top-right-bottom format, contains left top and right bottom corners
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Ltrb;
impl BBoxFormat for Ltrb {}

/// X-y-width-height format, contains coordinates of the center of bbox and width-height
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Xywh;
impl BBoxFormat for Xywh {}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BBox<F: BBoxFormat + Serialize + Deserialize<'static> + PartialEq>(
    [f32; 4],
    PhantomData<F>,
);

impl<F: BBoxFormat + Serialize + Deserialize<'static> + PartialEq> From<BBox<F>> for [f32; 4] {
    fn from(bbox: BBox<F>) -> Self {
        bbox.0
    }
}

impl<F: BBoxFormat + Serialize + Deserialize<'static> + PartialEq> BBox<F> {
    #[inline]
    pub fn as_slice(&self) -> &[f32; 4] {
        &self.0
    }

    // Use carefully when you REALLY sure that slice have needed format
    #[inline(always)]
    pub fn assigned(slice: &[f32; 4]) -> Self {
        BBox(*slice, Default::default())
    }
}

impl BBox<Ltwh> {
    #[inline]
    pub fn ltwh(x1: f32, x2: f32, x3: f32, x4: f32) -> Self {
        BBox([x1, x2, x3, x4], Default::default())
    }

    #[inline(always)]
    pub fn left(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn top(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn width(&self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub fn height(&self) -> f32 {
        self.0[3]
    }

    #[inline]
    pub fn as_xyah(&self) -> BBox<Xyah> {
        self.into()
    }

    #[inline]
    pub fn as_ltrb(&self) -> BBox<Ltrb> {
        self.into()
    }
}

impl BBox<Ltrb> {
    #[inline]
    pub fn ltrb(x1: f32, x2: f32, x3: f32, x4: f32) -> Self {
        BBox([x1, x2, x3, x4], Default::default())
    }

    #[inline]
    pub fn as_ltwh(&self) -> BBox<Ltwh> {
        self.into()
    }

    #[inline]
    pub fn as_xyah(&self) -> BBox<Xyah> {
        self.into()
    }

    #[inline(always)]
    pub fn left(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn top(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn right(&self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub fn bottom(&self) -> f32 {
        self.0[3]
    }
}

impl BBox<Xyah> {
    #[inline]
    pub fn xyah(x1: f32, x2: f32, x3: f32, x4: f32) -> Self {
        BBox([x1, x2, x3, x4], Default::default())
    }

    #[inline(always)]
    pub fn as_ltrb(&self) -> BBox<Ltrb> {
        self.into()
    }

    #[inline(always)]
    pub fn as_ltwh(&self) -> BBox<Ltwh> {
        self.into()
    }

    #[inline(always)]
    pub fn cx(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn cy(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn aspect_ratio(&self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub fn height(&self) -> f32 {
        self.0[3]
    }
}

impl BBox<Xywh> {
    #[inline]
    pub fn xywh(x1: f32, x2: f32, x3: f32, x4: f32) -> Self {
        BBox([x1, x2, x3, x4], Default::default())
    }

    #[inline(always)]
    pub fn as_xyah(&self) -> BBox<Xyah> {
        self.into()
    }

    #[inline(always)]
    pub fn cx(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn cy(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn width(&self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub fn height(&self) -> f32 {
        self.0[3]
    }
}

impl<'a> From<&'a BBox<Ltwh>> for BBox<Xyah> {
    #[inline]
    fn from(v: &'a BBox<Ltwh>) -> Self {
        Self(
            [
                v.0[0] + v.0[2] / 2.0,
                v.0[1] + v.0[3] / 2.0,
                v.0[2] / v.0[3],
                v.0[3],
            ],
            Default::default(),
        )
    }
}

impl<'a> From<&'a BBox<Ltrb>> for BBox<Xyah> {
    #[inline]
    fn from(v: &'a BBox<Ltrb>) -> Self {
        Self(
            [
                v.0[0] + (v.0[2] - v.0[0]) / 2.0,
                v.0[1] + (v.0[3] - v.0[1]) / 2.0,
                (v.0[2] - v.0[0]) / (v.0[3] - v.0[1]),
                v.0[3] - v.0[1],
            ],
            Default::default(),
        )
    }
}

impl<'a> From<&'a BBox<Ltwh>> for BBox<Ltrb> {
    #[inline]
    fn from(v: &'a BBox<Ltwh>) -> Self {
        Self(
            [v.0[0], v.0[1], v.0[2] + v.0[0], v.0[3] + v.0[1]],
            Default::default(),
        )
    }
}

impl<'a> From<&'a BBox<Xyah>> for BBox<Ltrb> {
    #[inline]
    fn from(v: &'a BBox<Xyah>) -> Self {
        Self(
            [
                v.0[0] - v.0[2] * v.0[3] / 2.,
                v.0[1] - v.0[3] / 2.,
                v.0[0] + v.0[2] * v.0[3] / 2.,
                v.0[1] + v.0[3] / 2.,
            ],
            Default::default(),
        )
    }
}

impl<'a> From<&'a BBox<Ltrb>> for BBox<Ltwh> {
    #[inline]
    fn from(v: &'a BBox<Ltrb>) -> Self {
        Self(
            [v.0[0], v.0[1], v.0[2] - v.0[0], v.0[3] - v.0[1]],
            Default::default(),
        )
    }
}

impl<'a> From<&'a BBox<Xyah>> for BBox<Ltwh> {
    #[inline]
    fn from(v: &'a BBox<Xyah>) -> Self {
        let height = v.0[3];
        let width = v.0[2] * height;

        Self(
            [v.0[0] - width / 2.0, v.0[1] - height / 2.0, width, height],
            Default::default(),
        )
    }
}

impl<'a> From<&'a BBox<Xywh>> for BBox<Xyah> {
    #[inline]
    fn from(v: &'a BBox<Xywh>) -> Self {
        Self(
            [v.0[0], v.0[1], v.0[2] / v.0[3], v.0[3]],
            Default::default(),
        )
    }
}

impl<'a> From<&'a BBox<Xyah>> for BBox<Xywh> {
    #[inline]
    fn from(v: &'a BBox<Xyah>) -> Self {
        Self(
            [v.0[0], v.0[1], v.0[2] * v.0[3], v.0[3]],
            Default::default(),
        )
    }
}
