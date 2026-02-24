use libm::sqrtf;

pub fn normalize4_or_default(v: [f32; 4], fallback: [f32; 4]) -> [f32; 4] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    if len_sq <= 1e-8 || !len_sq.is_finite() {
        return fallback;
    }
    let inv_len = 1.0 / sqrtf(len_sq);
    [
        v[0] * inv_len,
        v[1] * inv_len,
        v[2] * inv_len,
        v[3] * inv_len,
    ]
}

pub fn distance4_sq(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    let dw = a[3] - b[3];
    dx * dx + dy * dy + dz * dz + dw * dw
}
