use nalgebra as na;
use num_traits::Float;

pub fn linear_ls<T: na::ComplexField + Float>(
    x: na::DVector<T>,
    y: na::DVector<T>,
) -> Option<na::Matrix2x1<T>> {
    let n = T::from(x.len()).unwrap();

    let s_x = x.sum() + T::from(f32::EPSILON).unwrap();
    let x2 = x.map(|x| x * x);
    let s_x2 = x2.sum() + T::from(f32::EPSILON).unwrap();
    let s_xy = x.zip_map(&y, |x, y| x * y).sum();
    let s_y = y.sum();

    let a = na::Matrix2::new(s_x2, s_x, s_x, n);
    let b = na::Matrix2x1::new(s_xy, s_y);

    let qr_result = a.qr();
    let qty = qr_result.q().transpose() * b;
    let beta_hat = qr_result.r().solve_upper_triangular(&qty)?;

    Some(beta_hat)
}

pub fn quadratic_ls<T: na::ComplexField + Float>(
    x: &na::DVector<T>,
    y: &na::DVector<T>,
) -> std::option::Option<na::Matrix3x1<T>> {
    let n = T::from(x.len()).unwrap();

    let s_x1 = x.sum();
    let x2 = x.map(|x| x * x);
    let s_x2 = x2.sum();
    let x3 = x2.zip_map(x, |a, b| a * b);
    let s_x3 = x3.sum();
    let x4 = x3.zip_map(x, |a, b| a * b);
    let s_x4 = x4.sum();
    let s_x2y = x2.zip_map(y, |x, y| x * y).sum();
    let s_xy = x.zip_map(y, |x, y| x * y).sum();
    let s_y = y.sum();

    let a = na::Matrix3::new(s_x4, s_x3, s_x2, s_x3, s_x2, s_x1, s_x2, s_x1, n);
    let b = na::Matrix3x1::new(s_x2y, s_xy, s_y);

    let qr_result = a.qr();
    let qty = qr_result.q().transpose() * b;

    qr_result.r().solve_upper_triangular(&qty)
}

pub fn gauss(x: f32, c: f32) -> f32 {
    (-((x * x) / (2.0 * c * c))).exp()
}

#[inline]
pub fn approx_b<F, FN: FnOnce(F) -> F + Copy>(l: F, a: F, fx: FN) -> F
where
    F: na::RealField + Float,
{
    let mut dx = l;

    let count = if dx < F::from(0.0001).unwrap() { 0 } else { 8 };

    for _ in 0..count {
        let b = a + dx;
        let nl = na::distance(&na::Point2::new(a, fx(a)), &na::Point2::new(b, fx(b)));
        let dl = l / nl;

        dx *= dl;
    }

    a + dx
}
