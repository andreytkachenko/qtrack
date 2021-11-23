use super::math::{approx_b, quadratic_ls};
use nalgebra as na;
use nalgebra::Normed;
use num_traits::Float;

#[derive(Debug)]
pub struct Predictor<F>
where
    F: na::RealField + Float,
{
    pub min_a: F,
    pub curvature: F,
    pub direction: na::Complex<F>,
    pub extremum: na::Complex<F>,
    pub mean: na::Complex<F>,
    pub variance: na::Complex<F>,
    pub has_linear: bool,
    pub has_quadratic: bool,
    pub use_quadratic: bool,
}

impl<F> Predictor<F>
where
    F: na::RealField + Float,
{
    pub fn new(use_quadratic: bool, min_a: F) -> Self {
        Self {
            min_a,
            curvature: F::zero(),
            direction: na::Complex::new(F::zero(), F::zero()),
            extremum: na::Complex::new(F::zero(), F::zero()),
            mean: na::Complex::new(F::zero(), F::zero()),
            variance: na::Complex::new(F::zero(), F::zero()),
            has_linear: false,
            has_quadratic: false,
            use_quadratic,
        }
    }

    pub fn reset(&mut self) {
        self.curvature = F::zero();
        self.direction = na::Complex::new(F::zero(), F::zero());
        self.extremum = na::Complex::new(F::zero(), F::zero());
        self.mean = na::Complex::new(F::zero(), F::zero());
        self.variance = na::Complex::new(F::zero(), F::zero());
        self.has_linear = false;
        self.has_quadratic = false;
    }

    pub fn predict_dist(&self, from: na::Point2<F>, dist: F) -> na::Point2<F> {
        if self.has_quadratic {
            let pt = na::Complex::new(from.x - self.extremum.re, from.y - self.extremum.im)
                * self.direction.conj();

            let fx = move |x| self.curvature * x * x;
            let x = approx_b(dist, pt.re, fx);
            let c = na::Complex::new(x, fx(x)) * self.direction + self.extremum;

            na::Point2::new(c.re, c.im)
        } else if self.has_linear {
            let offset = self.direction * dist;

            from + na::Vector2::new(offset.re, offset.im)
        } else {
            from
        }
    }

    fn project_real(&self, pt: na::Complex<F>) -> (na::Complex<F>, F) {
        let mut curr = na::Complex::new(pt.re, self.curvature * pt.re * pt.re);
        let mut dist = pt.im - curr.im;

        let epsilon = F::from(0.000001).unwrap();
        let two = F::from(2).unwrap();

        for _ in 0..8 {
            println!();

            let tan = two * (curr.im / (curr.re + epsilon));
            let c = na::Unit::new_normalize(na::Complex::new(F::one(), tan)).into_inner();
            let proj = (pt - curr) * c.conj();

            dist = proj.im;
            let proj = na::Complex::new(proj.re, F::zero()) * c + curr;

            curr = na::Complex::new(proj.re, self.curvature * proj.re * proj.re);
        }

        (curr, dist)
    }

    fn project_image(&self, pt: na::Complex<F>) -> (na::Complex<F>, F) {
        let f = na::Complex::new(F::zero(), F::one() / self.curvature);
        let d = pt - f;
        let (a, b) = (d.im / d.re, f.im);

        let four = F::from(4).unwrap();
        let two = F::from(2).unwrap();

        let p1 = -(Float::sqrt(a * a + four * self.curvature * b) - a) / (two * self.curvature);
        let p2 = (Float::sqrt(a * a + four * self.curvature * b) + a) / (two * self.curvature);

        let p1 = na::Complex::new(p1, self.curvature * p1 * p1);
        let p2 = na::Complex::new(p2, self.curvature * p2 * p2);

        let d1 = (p1 - pt).norm();
        let d2 = (p2 - pt).norm();

        if d1 < d2 {
            (p1, -d1)
        } else {
            (p2, -d2)
        }
    }

    pub fn project(&self, pt: na::Point2<F>) -> (na::Point2<F>, F) {
        if self.has_quadratic {
            let mut pt = na::Complex::new(pt.x - self.extremum.re, pt.y - self.extremum.im)
                * self.direction.conj();

            let epsilon = F::from(0.001).unwrap();

            if Float::abs(pt.re) < epsilon {
                if pt.re.is_negative() {
                    pt.re = -epsilon;
                } else {
                    pt.re = epsilon;
                }
            }

            let (curr, dist) = if pt.im.is_sign_positive() == self.curvature.is_sign_positive() {
                self.project_real(pt)
            } else {
                self.project_image(pt)
            };

            let curr = curr * self.direction + self.extremum;

            (na::Point2::new(curr.re, curr.im), dist)
        } else if self.has_linear {
            let npt =
                na::Complex::new(pt.x - self.mean.re, pt.y - self.mean.im) * self.direction.conj();
            let dist = npt.im;
            let npt = na::Complex::new(npt.re, F::zero()) * self.direction;

            (
                na::Point2::new(npt.re + self.mean.re, npt.im + self.mean.im),
                dist,
            )
        } else {
            (
                pt,
                na::distance(&pt, &na::Point2::new(self.mean.re, self.mean.im)),
            )
        }
    }

    pub fn update<'a, I: IntoIterator<Item = (&'a na::Point2<F>, F)>>(&mut self, points: I) {
        self.has_linear = false;
        self.has_quadratic = false;
        self.curvature = F::zero();

        // Calculating Mean

        let iter = points.into_iter();
        let mut x = Vec::with_capacity(iter.size_hint().0);
        let mut y = Vec::with_capacity(iter.size_hint().0);
        let mut w = Vec::with_capacity(iter.size_hint().0);

        self.mean = na::Complex::new(F::zero(), F::zero());
        self.variance = na::Complex::new(F::zero(), F::zero());
        let mut wsum = F::zero();
        let mut dir = na::Complex::new(F::zero(), F::zero());
        let mut prev = na::Complex::new(F::zero(), F::zero());

        for (idx, (p, w_)) in iter.enumerate() {
            if idx == 0 {
                prev = na::Complex::new(p.x, p.y);
            } else {
                let pt = na::Complex::new(p.x, p.y);
                dir += na::Unit::new_normalize(prev - pt).into_inner();
                prev = pt;
            }

            x.push(p.x);
            y.push(p.y);
            w.push(w_);

            self.mean.re += p.x * w_;
            self.mean.im += p.y * w_;

            wsum += w_;
        }

        self.mean /= wsum;
        dir = na::Unit::new_normalize(dir).into_inner();

        let n = x.len();
        let (x, y): (na::DVector<F>, na::DVector<F>) = (x.into(), y.into());
        let var = (x.variance(), y.variance());

        if n < 3 || Float::max(var.0, var.1) < F::from(25).unwrap() {
            return;
        }

        self.direction = dir;
        self.has_linear = true;

        // Fitting Polyline

        if !self.use_quadratic || x.len() < 8 {
            return;
        }

        let tr = self.direction.conj();
        let (mut x, mut y) = (x, y);
        // let qmean = na::Complex::new(x.mean(), y.mean());

        x.iter_mut().zip(y.iter_mut()).for_each(|(x, y)| {
            let c = (na::Complex::new(*x, *y) - self.mean) * tr;

            *x = c.re;
            *y = c.im;
        });

        if x.variance() < F::from(45).unwrap() {
            return;
        }

        let params = if let Some(m) = quadratic_ls(&x, &y) {
            m
        } else {
            return;
        };

        if Float::abs(params[0]) < self.min_a {
            return;
        }

        let x0 = -params[1] / (F::from(2).unwrap() * params[0]);
        let y0 = params[0] * x0 * x0 + params[1] * x0 + params[2];

        self.extremum = self.mean + na::Complex::new(x0, y0) * self.direction;
        self.curvature = params[0] * F::from(0.75).unwrap();
        self.has_quadratic = true;
    }
}
