use std::iter;
use nalgebra as na;

fn plot_2d(spline: &bspline::BSpline<na::Vector2<f32>, f32>, plot: &mut [u8], plot_dim: (usize, usize), scale: (f32, f32),
           offset: (f32, f32)) {
    let step_size = 0.001;
    let t_range = spline.knot_domain();
    let steps = ((t_range.1 - t_range.0) / step_size) as usize;
    for s in 0..steps + 1 {
        let t = step_size * s as f32 + t_range.0;
        let pt = spline.point(t);
        let ix = ((pt.x + offset.0) * scale.0) as isize;
        let iy = ((pt.y + offset.1) * scale.1) as isize;
        for y in iy - 1..iy + 1 {
            for x in ix - 1..ix + 1 {
                if y >= 0 && y < plot_dim.1 as isize && x >= 0 && x < plot_dim.0 as isize {
                    let px = (plot_dim.1 - 1 - y as usize) * plot_dim.0 * 3 + x as usize * 3;
                    for i in 0..3 {
                        plot[px + i] = 0;
                    }
                }
            }
        }
    }
    // Draw the control points
    for pt in spline.control_points() {
        let ix = ((pt.x + offset.0) * scale.0) as isize;
        let iy = ((pt.y + offset.1) * scale.1) as isize;
        for y in iy - 3..iy + 3 {
            for x in ix - 3..ix + 3 {
                if y >= 0 && y < plot_dim.1 as isize && x >= 0 && x < plot_dim.0 as isize {
                    let px = (plot_dim.1 - 1 - y as usize) * plot_dim.0 * 3 + x as usize * 3;
                    plot[px] = 255;
                    plot[px + 1] = 0;
                    plot[px + 2] = 0;
                }
            }
        }
    }
}


fn main() {
    let points = vec![
        na::Vector2::new(-1.5, -1.5), na::Vector2::new(-0.5, 1.5),
        na::Vector2::new(0.5, -1.5), na::Vector2::new(1.5, 1.5)
    ];

    // let knots = vec![0.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 7.0];
    let knots = vec![0.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0];
    
    let degree = 3;

    let plot_dim = (720, 540);
    let scale = (plot_dim.0 as f32 / 4.0, plot_dim.1 as f32 / 4.0);
    let offset = (2.0, 2.0);

    let mut plot: Vec<_> = iter::repeat(255u8).take(plot_dim.0 * plot_dim.1 * 3).collect();

    println!("Plotting Cubic B-spline with:\n\tpoints = {:?}\n\tknots = {:?}", points, knots);

    let spline = bspline::BSpline::new(degree, points, knots);

    println!("\tt range = {:?}", spline.knot_domain());
    println!("spline = {:?}", spline);

    plot_2d(&spline, &mut plot[..], plot_dim, scale, offset);

    match image::save_buffer("cubic_2d.png", &plot[..], plot_dim.0 as u32, plot_dim.1 as u32, image::ColorType::Rgb8) {
        Ok(_) => println!("2D Cubic B-spline saved to cubic_2d.png"),
        Err(e) => println!("Error saving cubic_2d.png,  {}", e),
    }
}