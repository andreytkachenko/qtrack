pub mod bbox;
pub mod detection;
pub mod detector;
pub mod error;
pub mod frame;
pub mod math;
pub mod rolling_avg;
pub mod scene;
pub mod tracker;

mod circular_queue;
mod predictor;
mod track;

pub use detection::Detection;
pub use frame::Frame;
pub use track::Track;

use error::Error;
use nalgebra as na;
use std::collections::HashMap;
use std::{fmt, rc::Rc};

pub trait Float:
    num_traits::FromPrimitive + na::ComplexField + Copy + fmt::Debug + PartialEq + 'static
{
}

impl<T> Float for T where
    T: num_traits::FromPrimitive + na::ComplexField + Copy + fmt::Debug + PartialEq + 'static
{
}

pub trait Tracking {
    fn predict(&mut self, src: &str);
    fn update(&mut self, dets: &[Frame], src: &str) -> Result<(), error::Error>;
    fn tracks(&self, src: &str) -> Rc<[Track]>;
}

pub struct QuadraticTracker {
    scenes: HashMap<String, scene::Scene>,
}

impl QuadraticTracker {
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
        }
    }
}

impl Default for QuadraticTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::Tracking for QuadraticTracker {
    #[inline]
    fn predict(&mut self, _src: &str) {}

    fn update(&mut self, frames: &[Frame], src: &str) -> Result<(), Error> {
        for frame in frames {
            let item = self.scenes.get_mut(src);
            let scene = if let Some(scene) = item {
                scene
            } else {
                let padding = 10.0;
                let (fw, fh) = frame.dims;
                let poly = vec![
                    na::Point2::new(padding, padding),
                    na::Point2::new(fw as f32 - padding - padding, padding),
                    na::Point2::new(fw as f32 - padding - padding, fh as f32 - padding - padding),
                    na::Point2::new(padding, fh as f32 - padding - padding),
                ];

                self.scenes
                    .entry(src.to_string())
                    .or_insert_with(|| scene::Scene::new(poly))
            };

            scene.update_time(frame.timestamp);
            let mapping = scene.map_detections(frame.timestamp, &frame.detections);
            scene.update(mapping);
        }
        Ok(())
    }

    #[inline]
    fn tracks(&self, src: &str) -> Rc<[Track]> {
        if let Some(ctx) = self.scenes.get(src) {
            return ctx.tracks().into_boxed_slice().into();
        }

        Rc::new([])
    }
}
