use super::math;
use super::{predictor::Predictor, rolling_avg::RollingAvg};
use crate::Detection;
use nalgebra as na;

#[derive(Debug)]
pub struct Object {
    pub predictor: Predictor<f32>,
    pub history: RollingAvg,
    pub prev_vel: f32,
    pub vel: f32,
    pub vel_updated_at: f32,
    pub pos: na::Point2<f32>,
    pub ts: f32,
    pub initialized: bool,
    pub correction: f32,
}

impl Object {
    pub fn new(md: f32, correction: f32, use_quadratic: bool) -> Self {
        Self {
            predictor: Predictor::new(use_quadratic, 0.00025),
            history: RollingAvg::new(md, (80.0 / md).round() as usize),
            pos: na::Point2::new(0.0, 0.0),
            prev_vel: 0.0,
            vel_updated_at: 0.0,
            vel: 0.0,
            ts: 0.0,
            initialized: false,
            correction,
        }
    }

    pub fn reset(&mut self) {
        self.predictor.reset();
        self.pos = na::Point2::new(0.0, 0.0);
        self.vel_updated_at = 0.0;
        self.vel = 0.0;
        self.prev_vel = 0.0;
        self.ts = 0.0;
        self.initialized = false;
        self.history.clear();
    }

    #[inline]
    pub fn predict_distance(&self, _id: u32, ts: f32) -> f32 {
        let pred_dist = self.vel * (ts - self.ts).max(0.033).abs();

        pred_dist + self.vel * self.correction
    }

    #[inline]
    pub fn predict(&self, id: u32, ts: f32) -> na::Point2<f32> {
        self.predictor
            .predict_dist(self.pos, self.predict_distance(id, ts))
    }

    #[inline]
    pub fn probability(&self, id: u32, ts: f32, pp: na::Point2<f32>, avg: &Detection) -> f32 {
        let size = avg.w.min(avg.h);
        let pred_dist = self.predict_distance(id, ts);

        let (pt, d) = self.predictor.project(pp);
        let (dir, mut dist) = na::Unit::new_and_get(pt - self.pos);
        let dir = na::Complex::new(dir.x, dir.y);
        let dt = dir * self.predictor.direction.conj();

        if dt.re < 0.0 {
            dist = -dist;
        }

        let gap = pred_dist * 0.5 + size * 0.6;
        let angle = math::gauss(d.abs(), (pred_dist / 4.0).max(17.0));
        let speed = math::gauss((pred_dist - dist.abs()).abs(), gap);

        angle * speed
    }

    pub fn update(&mut self, id: u32, ts: f32, pp: na::Point2<f32>, rp: na::Point2<f32>) {
        if self.history.push(ts, rp) {
            self.predictor.update(self.history.iter_points());
        }

        if let Some(vel) = self.history.velocity() {
            self.prev_vel = self.vel;

            if self.initialized {
                // self.vel = self.vel * 0.9 + vel * 0.1;

                let dt = ts - self.vel_updated_at;
                let dvel = (vel - self.vel) / dt;
                let dvel = dvel.clamp(-10_000.0, 10_000.0);

                // println!("{}: {}", id, dvel);

                self.vel += (dvel * dt) * 0.2;
                self.vel_updated_at = ts;
            } else {
                self.vel = vel;
                self.initialized = true;
            }
        }

        // if self.vel < 5.0 {
        //     self.predictor.has_linear = false;
        //     self.predictor.has_quadratic = false;
        // }

        let (pos, _) = self.predictor.project(pp);

        if self.predictor.has_quadratic || self.predictor.has_linear {
            self.pos = (pos.coords * 0.2 + self.predict(id, ts).coords * 0.8).into();
        } else if self.initialized {
            self.pos = (self.pos.coords * 0.9 + pos.coords * 0.1).into();
        } else {
            self.pos = pos;
        }

        self.ts = ts;
    }
}
