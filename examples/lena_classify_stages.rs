use std::fs::File;
use rsface::haar::Cascade;
use rsface::image::GrayImage;
use rsface::detector::DetectorConfig;
use rsface::integral::IntegralImage;
use rsface::haar::EvalCache;

fn main() {
    let mut f = File::open("/tmp/cascade_eval/lena.pgm").expect("open");
    let img: GrayImage = rsface::image::codec::read_pgm(&mut f).expect("read_pgm");
    let mut eq = img.clone();
    eq.equalize_hist_inplace();

    let cascade = Cascade::load(std::path::Path::new("cascade.rfcf")).expect("load");
    let ii = IntegralImage::from_gray(&eq);
    let ri = rsface::integral::RotatedIntegralImage::from_gray(&eq);
    let sq = rsface::integral::SquaredIntegralImage::from_gray(&eq);
    let mut cache = EvalCache::new(cascade.features.len());
    cache.set_squared_iis(sq);

    let (x, y) = (220, 220);
    let ww = cascade.window_w;
    let wh = cascade.window_h;
    let nw = ww - 2;
    let nh = wh - 2;
    let nx1 = x + 1;
    let ny1 = y + 1;
    let nx2 = nx1 + nw;
    let ny2 = ny1 + nh;
    let nw_area = (nw as f64) * (nh as f64);
    let sum_in = ii.rect_sum(nx1, ny1, nx2, ny2);
    let sum_sq_in = cache.sum_sq_rect_sum(nx1, ny1, nx2, ny2);
    let variance_part = nw_area * (sum_sq_in as f64) - (sum_in as f64) * (sum_in as f64);
    let variance_norm_factor: f32 = if variance_part > 0.0 { (1.0 / variance_part.sqrt()) as f32 } else { 0.0 };
    println!("variance_norm_factor={:.8} (part={})", variance_norm_factor, variance_part);

    cache.clear();
    for (si, stage) in cascade.stages.iter().take(3).enumerate() {
        let mut stage_sum: f32 = 0.0;
        println!("--- stage {} threshold={:.3} ({} features) ---", si, stage.stage_threshold, stage.weak_features.len());
        for (wi, w) in stage.weak_features.iter().enumerate() {
            let raw = cache.get_or_eval(
                w.feature_index as usize,
                &cascade.features[w.feature_index as usize],
                &ii, &ri, x, y, ww, wh, ii.width(), ii.height());
            let value = raw * variance_norm_factor;
            let chosen = if value < w.threshold { w.left_val } else { w.right_val };
            stage_sum += chosen;
            if si < 3 && wi < 5 {
                println!("  s{} w{}: feat#{} raw={:.0} norm*={:.5} thr={:.4} -> val={:.3}",
                    si, wi, w.feature_index, raw, value, w.threshold, chosen);
            }
        }
        println!("stage {} sum={:.3} thr={:.3} -> {}", si, stage_sum, stage.stage_threshold,
            if stage_sum >= stage.stage_threshold { "PASS" } else { "FAIL" });
        if stage_sum < stage.stage_threshold { return; }
    }
    let _ = DetectorConfig::default();
}
