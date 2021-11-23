use crate::circular_queue::CircularQueue;
use nalgebra as na;

#[derive(Debug, Clone)]
pub struct RollingAvg {
    min_dist: f32,
    curr: (f32, na::Point2<f32>),
    history: CircularQueue<(f32, na::Point2<f32>, f32)>,
}

impl RollingAvg {
    pub fn new(min_dist: f32, hcount: usize) -> Self {
        Self {
            min_dist,
            curr: (0.0, na::Point2::new(0.0, 0.0)),
            history: CircularQueue::with_capacity(hcount),
        }
    }

    pub fn clear(&mut self) {
        self.curr = (0.0, na::Point2::new(0.0, 0.0));
        self.history.clear();
    }

    pub fn push(&mut self, ts: f32, pos: na::Point2<f32>) -> bool {
        if na::distance(&self.curr.1, &pos) < self.min_dist && ts - self.curr.0 < 5. {
            if let Some(top) = self.history.iter_mut().next() {
                let weight = top.2;
                top.0 = (top.0 * weight + ts) / (weight + 1.0);
                top.1 = ((top.1.coords * weight + pos.coords) / (weight + 1.0)).into();
                top.2 += 1.0;

                return false;
            }
        }

        self.history.push((ts, pos, 1.0));
        self.curr = (ts, pos);

        loop {
            let update = {
                let mut iter = self.history.iter();
                if let (Some(curr), Some(top), Some(prev)) = (iter.next(), iter.next(), iter.next())
                {
                    let q1 = na::Unit::new_normalize(top.1.coords - curr.1.coords);
                    let q2 = na::Unit::new_normalize(prev.1.coords - top.1.coords);

                    q1.dot(&q2) < -0.0
                } else {
                    false
                }
            };

            if update {
                self.history.pop();

                if let Some(top) = self.history.top_mut() {
                    let weight = top.2;
                    top.0 = (top.0 * weight + ts) / (weight + 1.0);
                    top.1 = ((top.1.coords * weight + pos.coords) / (weight + 1.0)).into();
                    top.2 += 1.0;
                }
            } else {
                break;
            }
        }

        true
    }

    pub fn velocity(&self) -> Option<f32> {
        let mut iter = self.history.iter();

        let top = iter.next()?;
        let prev = iter.next()?;

        let dl = na::distance(&top.1, &prev.1);
        let dt = top.0 - prev.0;

        Some(dl / dt)
    }

    #[inline]
    pub fn top(&self) -> Option<&(f32, na::Point2<f32>, f32)> {
        self.history.iter().next()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &(f32, na::Point2<f32>, f32)> {
        self.history.iter()
    }

    #[inline]
    pub fn iter_points(&self) -> impl Iterator<Item = (&na::Point2<f32>, f32)> {
        self.history.iter().map(|(_, x, w)| (x, *w))
    }

    #[inline]
    pub fn num_points(&self) -> usize {
        self.history.len()
    }
}
