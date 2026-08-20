use rsface::haar::Cascade;

fn main() {
    let cascade = Cascade::load(std::path::Path::new("cascade.rfcf")).expect("load");
    println!("Window: {}x{}, {} stages, {} features", cascade.window_w, cascade.window_h, cascade.stages.len(), cascade.features.len());
    // Dump first 3 features and first 3 stages
    for (i, f) in cascade.features.iter().take(3).enumerate() {
        println!("Feature {}: kind={:?} {}x{}", i, f.kind, f.width, f.height);
        for r in &f.rects {
            println!("  rect: x={} y={} {}x{} weight={}", r.x, r.y, r.w, r.h, r.weight);
        }
    }
    for (i, s) in cascade.stages.iter().take(3).enumerate() {
        println!("Stage {}: threshold={:.4}, {} features", i, s.stage_threshold, s.weak_features.len());
        for w in s.weak_features.iter().take(3) {
            println!("  weak: feat#{} thr={:.4} left={:.4} right={:.4}", w.feature_index, w.threshold, w.left_val, w.right_val);
        }
    }
}
