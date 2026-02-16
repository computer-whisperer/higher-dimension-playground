const DEFAULT_SYNTH_SEED: u64 = 1234;
const DITHER_SEED: u64 = 777;
const BOX_MULLER_MIN_U1: f32 = 1e-12;
const MIN_FILTER_CUTOFF_HZ: f32 = 1.0;
const MAX_FILTER_CUTOFF_NYQUIST_SCALE: f32 = 0.95;
const GRAIN_WINDOW_REFERENCE_SAMPLES: f32 = 600.0;
const REFERENCE_SAMPLE_RATE: f32 = 44_100.0;

pub struct SynthesizedEffects {
    pub place: Vec<f32>,
    pub break_clip: Vec<f32>,
    pub footsteps: [Vec<f32>; 3],
    pub jump: Vec<f32>,
    pub land: Vec<f32>,
}

pub fn synthesize_effects(sample_rate: u32) -> SynthesizedEffects {
    assert!(sample_rate > 0);
    let sr = sample_rate as f32;
    let mut rng = Rng::new(DEFAULT_SYNTH_SEED);

    let place_click_a = gen_click(&mut rng, sample_rate, 0.06, 1600.0, 0.02, 0.9);
    let place_click_b = gen_click(&mut rng, sample_rate, 0.03, 2600.0, 0.01, 0.35);
    let place_block = mix(&[&place_click_a, &place_click_b]);

    let break_block = gen_crunch(&mut rng, sample_rate, 0.18, 1.0);

    let footstep_1_f0 = 95.0 + rng.f32_range(-8.0, 8.0);
    let footstep_1_f1 = 60.0 + rng.f32_range(-6.0, 6.0);
    let footstep_1_thud = gen_thud(
        &mut rng,
        sample_rate,
        0.10,
        footstep_1_f0,
        footstep_1_f1,
        0.9,
    );
    let footstep_1_crunch = gen_crunch(&mut rng, sample_rate, 0.06, 0.25);
    let footstep_1 = mix(&[&footstep_1_thud, &footstep_1_crunch]);

    let footstep_2_f0 = 95.0 + rng.f32_range(-8.0, 8.0);
    let footstep_2_f1 = 60.0 + rng.f32_range(-6.0, 6.0);
    let footstep_2_thud = gen_thud(
        &mut rng,
        sample_rate,
        0.10,
        footstep_2_f0,
        footstep_2_f1,
        0.9,
    );
    let footstep_2_crunch = gen_crunch(&mut rng, sample_rate, 0.06, 0.25);
    let footstep_2 = mix(&[&footstep_2_thud, &footstep_2_crunch]);

    let footstep_3_f0 = 95.0 + rng.f32_range(-8.0, 8.0);
    let footstep_3_f1 = 60.0 + rng.f32_range(-6.0, 6.0);
    let footstep_3_thud = gen_thud(
        &mut rng,
        sample_rate,
        0.10,
        footstep_3_f0,
        footstep_3_f1,
        0.9,
    );
    let footstep_3_crunch = gen_crunch(&mut rng, sample_rate, 0.06, 0.25);
    let footstep_3 = mix(&[&footstep_3_thud, &footstep_3_crunch]);

    let jump_whoosh = gen_whoosh(&mut rng, sample_rate, 0.16, 0.7);
    let jump_click = gen_click(&mut rng, sample_rate, 0.03, 2300.0, 0.012, 0.25);
    let jump = mix(&[&jump_whoosh, &jump_click]);

    let land_thud = gen_thud(&mut rng, sample_rate, 0.16, 95.0, 45.0, 1.0);
    let land_crunch = gen_crunch(&mut rng, sample_rate, 0.10, 0.35);
    let land = mix(&[&land_thud, &land_crunch]);

    let mut dither_rng = Rng::new(DITHER_SEED);
    let place = process_medium(place_block, SoundClass::UiLike, sr, &mut dither_rng);
    let break_clip = process_medium(break_block, SoundClass::Body, sr, &mut dither_rng);
    let footstep_1 = process_medium(footstep_1, SoundClass::Body, sr, &mut dither_rng);
    let footstep_2 = process_medium(footstep_2, SoundClass::Body, sr, &mut dither_rng);
    let footstep_3 = process_medium(footstep_3, SoundClass::Body, sr, &mut dither_rng);
    let jump = process_medium(jump, SoundClass::Jump, sr, &mut dither_rng);
    let land = process_medium(land, SoundClass::Body, sr, &mut dither_rng);

    SynthesizedEffects {
        place,
        break_clip,
        footsteps: [footstep_1, footstep_2, footstep_3],
        jump,
        land,
    }
}

#[derive(Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 32) as u32
    }

    fn f32(&mut self) -> f32 {
        let v = self.next_u32();
        (v as f32) / (u32::MAX as f32 + 1.0)
    }

    fn f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    fn i_range(&mut self, lo: i32, hi_inclusive: i32) -> i32 {
        let span = (hi_inclusive - lo + 1) as u32;
        lo + (self.next_u32() % span) as i32
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.f32().max(BOX_MULLER_MIN_U1) as f64;
        let u2 = self.f32() as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos()) as f32
    }
}

fn onepole_lowpass(x: &[f32], cutoff_hz: f32, sr: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let nyquist = 0.5 * sr;
    let cutoff = cutoff_hz.clamp(
        MIN_FILTER_CUTOFF_HZ,
        nyquist * MAX_FILTER_CUTOFF_NYQUIST_SCALE,
    );
    let dt = 1.0 / sr;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    let a = dt / (rc + dt);
    let mut y = Vec::with_capacity(x.len());
    let mut prev = a * x[0];
    y.push(prev);
    for &xi in x.iter().skip(1) {
        prev = prev + a * (xi - prev);
        y.push(prev);
    }
    y
}

fn onepole_highpass(x: &[f32], cutoff_hz: f32, sr: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let nyquist = 0.5 * sr;
    let cutoff = cutoff_hz.clamp(
        MIN_FILTER_CUTOFF_HZ,
        nyquist * MAX_FILTER_CUTOFF_NYQUIST_SCALE,
    );
    let dt = 1.0 / sr;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    let a = rc / (rc + dt);
    let mut y = Vec::with_capacity(x.len());
    let mut prev_y = x[0];
    let mut prev_x = x[0];
    y.push(prev_y);
    for &xi in x.iter().skip(1) {
        let yi = a * (prev_y + xi - prev_x);
        y.push(yi);
        prev_y = yi;
        prev_x = xi;
    }
    y
}

fn env_adsr(n: usize, sr: f32, a_s: f32, d_s: f32, sustain: f32, r_s: f32) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }

    let mut na = (a_s * sr) as usize;
    let mut nd = (d_s * sr) as usize;
    let mut nr = (r_s * sr) as usize;
    if na + nd + nr > n {
        let total = (na + nd + nr).max(1);
        let scale = n as f32 / total as f32;
        na = (na as f32 * scale) as usize;
        nd = (nd as f32 * scale) as usize;
        nr = n.saturating_sub(na + nd);
    }
    let ns = n.saturating_sub(na + nd + nr);

    let mut e = vec![0.0f32; n];
    let mut idx = 0;

    if na > 0 {
        for i in 0..na {
            e[idx + i] = i as f32 / na as f32;
        }
        idx += na;
    }
    if nd > 0 {
        for i in 0..nd {
            let t = i as f32 / nd as f32;
            e[idx + i] = 1.0 + t * (sustain - 1.0);
        }
        idx += nd;
    }
    if ns > 0 {
        for i in 0..ns {
            e[idx + i] = sustain;
        }
        idx += ns;
    }
    if nr > 0 && idx < n {
        let end = (idx + nr).min(n);
        let len = end - idx;
        for i in 0..len {
            let t = if len <= 1 {
                1.0
            } else {
                i as f32 / (len - 1) as f32
            };
            e[idx + i] = sustain * (1.0 - t);
        }
    }
    e
}

fn mix(signals: &[&[f32]]) -> Vec<f32> {
    let n = signals.iter().map(|s| s.len()).max().unwrap_or(0);
    if n == 0 {
        return Vec::new();
    }
    let mut y = vec![0.0f32; n];
    for signal in signals {
        for (i, &v) in signal.iter().enumerate() {
            y[i] += v;
        }
    }
    let peak = y.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak > 1e-9 {
        let scale = 1.0 / (peak * 1.05).max(1.0);
        for sample in &mut y {
            *sample *= scale;
        }
    }
    y
}

fn normalize_peak(mut x: Vec<f32>, target: f32) -> Vec<f32> {
    let peak = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak > 1e-9 {
        let scale = target / peak;
        for sample in &mut x {
            *sample *= scale;
        }
    }
    x
}

fn tpdf_dither(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    let mut dither = vec![0.0f32; n];
    for sample in &mut dither {
        *sample = (rng.f32() - rng.f32()) * amp;
    }
    dither
}

fn duration_samples(duration_s: f32, sample_rate: u32) -> usize {
    (duration_s * sample_rate as f32).max(0.0) as usize
}

fn gen_click(
    rng: &mut Rng,
    sample_rate: u32,
    duration_s: f32,
    pitch_hz: f32,
    decay_s: f32,
    amp: f32,
) -> Vec<f32> {
    let n = duration_samples(duration_s, sample_rate);
    let sr = sample_rate as f32;
    let mut y = vec![0.0f32; n];
    for (i, sample) in y.iter_mut().enumerate() {
        let t = i as f32 / sr;
        let tone = (2.0 * std::f32::consts::PI * pitch_hz * t).sin() * (-t / decay_s).exp();
        let noise = rng.normal() * (-t / (decay_s * 0.8)).exp();
        *sample = amp * (0.7 * tone + 0.3 * noise);
    }
    y
}

fn gen_thud(
    rng: &mut Rng,
    sample_rate: u32,
    duration_s: f32,
    f0: f32,
    f1: f32,
    amp: f32,
) -> Vec<f32> {
    let n = duration_samples(duration_s, sample_rate);
    let sr = sample_rate as f32;

    let mut phase = 0.0f32;
    let mut tone = vec![0.0f32; n];
    for (i, sample) in tone.iter_mut().enumerate() {
        let f = f0 + (f1 - f0) * (i as f32 / n.max(1) as f32);
        phase += 2.0 * std::f32::consts::PI * f / sr;
        *sample = phase.sin();
    }

    let mut noise = vec![0.0f32; n];
    for sample in &mut noise {
        *sample = rng.normal();
    }
    let noise = onepole_lowpass(&noise, 600.0, sr);
    let envelope = env_adsr(n, sr, 0.002, 0.03, 0.0, 0.10);

    let mut y = vec![0.0f32; n];
    for i in 0..n {
        y[i] = amp * (0.75 * tone[i] + 0.25 * noise[i]) * envelope[i];
    }
    y
}

fn gen_crunch(rng: &mut Rng, sample_rate: u32, duration_s: f32, amp: f32) -> Vec<f32> {
    let n = duration_samples(duration_s, sample_rate);
    let sr = sample_rate as f32;

    let mut base = vec![0.0f32; n];
    let mut base2 = vec![0.0f32; n];
    for i in 0..n {
        base[i] = rng.normal();
        base2[i] = rng.normal();
    }

    let band = onepole_lowpass(&onepole_highpass(&base, 200.0, sr), 3000.0, sr);
    let band2 = onepole_lowpass(&onepole_highpass(&base2, 600.0, sr), 8000.0, sr);
    let envelope = env_adsr(n, sr, 0.001, 0.05, 0.0, 0.10);

    let mut y = vec![0.0f32; n];
    for i in 0..n {
        y[i] = (0.7 * band[i] + 0.3 * band2[i]) * envelope[i];
    }

    let grain_window = ((GRAIN_WINDOW_REFERENCE_SAMPLES / REFERENCE_SAMPLE_RATE)
        * sample_rate as f32)
        .round() as i32;
    for _ in 0..8 {
        let pos = rng.i_range(0, (n as i32 - grain_window.max(1)).max(0)) as usize;
        let pitch = rng.f32_range(900.0, 2800.0);
        let grain = gen_click(rng, sample_rate, 0.012, pitch, 0.006, 0.25);
        let end = (pos + grain.len()).min(n);
        for i in 0..(end - pos) {
            y[pos + i] += grain[i];
        }
    }

    let y = onepole_lowpass(&y, 6000.0, sr);
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = amp * y[i];
    }
    out
}

fn gen_whoosh(rng: &mut Rng, sample_rate: u32, duration_s: f32, amp: f32) -> Vec<f32> {
    let n = duration_samples(duration_s, sample_rate);
    let sr = sample_rate as f32;

    let mut noise = vec![0.0f32; n];
    for sample in &mut noise {
        *sample = rng.normal();
    }

    let mut y = vec![0.0f32; n];
    let chunk = 256usize;
    for i in (0..n).step_by(chunk) {
        let cutoff = 800.0 + (4200.0 - 800.0) * (i as f32 / n.max(1) as f32);
        let end = (i + chunk).min(n);
        let filtered = onepole_lowpass(&noise[i..end], cutoff, sr);
        for j in 0..(end - i) {
            y[i + j] = filtered[j];
        }
    }

    let envelope = env_adsr(n, sr, 0.01, 0.05, 0.2, 0.08);
    let y = onepole_highpass(&y, 120.0, sr);
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = amp * y[i] * envelope[i];
    }
    out
}

fn soften_medium(
    x: &[f32],
    sr: f32,
    hf_cut: f32,
    hf_atten: f32,
    top_lp: f32,
    transient_soft: f32,
) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let mean = x.iter().copied().sum::<f32>() / x.len() as f32;
    let mut x0 = x.iter().map(|v| v - mean).collect::<Vec<_>>();
    x0 = onepole_highpass(&x0, 30.0, sr);

    let low = onepole_lowpass(&x0, hf_cut, sr);
    let mut high = x0
        .iter()
        .zip(low.iter())
        .map(|(full, low_band)| full - low_band)
        .collect::<Vec<_>>();

    if transient_soft > 0.0 {
        let mut w = ((transient_soft * 0.002 * sr).round() as usize).max(3);
        if w % 2 == 0 {
            w += 1;
        }
        let half = w / 2;
        let mut smoothed = vec![0.0f32; high.len()];
        for i in 0..high.len() {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(high.len());
            let mut acc = 0.0;
            for sample in &high[lo..hi] {
                acc += *sample;
            }
            smoothed[i] = acc / (hi - lo).max(1) as f32;
        }
        for i in 0..high.len() {
            high[i] = 0.65 * smoothed[i] + 0.35 * high[i];
        }
    }

    for sample in &mut high {
        *sample *= hf_atten;
    }
    high = onepole_lowpass(&high, top_lp, sr);

    let mut y = vec![0.0f32; x.len()];
    for i in 0..x.len() {
        y[i] = low[i] + high[i];
    }
    onepole_lowpass(&y, 7800.0, sr)
}

#[derive(Clone, Copy)]
enum SoundClass {
    UiLike,
    Jump,
    Body,
}

impl SoundClass {
    fn params(self) -> (f32, f32, f32, f32, f32) {
        match self {
            Self::UiLike => (4200.0, 0.60, 9000.0, 0.12, 0.82),
            Self::Jump => (4000.0, 0.65, 9000.0, 0.08, 0.85),
            Self::Body => (3600.0, 0.58, 8500.0, 0.10, 0.92),
        }
    }
}

fn process_medium(mut x: Vec<f32>, class: SoundClass, sr: f32, dither_rng: &mut Rng) -> Vec<f32> {
    let (hf_cut, hf_atten, top_lp, transient_soft, peak) = class.params();
    x = soften_medium(&x, sr, hf_cut, hf_atten, top_lp, transient_soft);
    x = normalize_peak(x, peak);

    let dither = tpdf_dither(dither_rng, x.len(), 1.0 / 32768.0);
    for i in 0..x.len() {
        x[i] = (x[i] + dither[i]).clamp(-1.0, 1.0);
    }

    // Match the old WAV path: dither, quantize to i16 PCM, then back to normalized f32.
    for sample in &mut x {
        let pcm = (*sample * i16::MAX as f32).round() as i16;
        *sample = pcm as f32 / i16::MAX as f32;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesized_effects_are_non_empty_and_bounded() {
        let effects = synthesize_effects(44_100);
        let clips = [
            &effects.place,
            &effects.break_clip,
            &effects.footsteps[0],
            &effects.footsteps[1],
            &effects.footsteps[2],
            &effects.jump,
            &effects.land,
        ];
        for clip in clips {
            assert!(!clip.is_empty());
            for &sample in clip.iter() {
                assert!((-1.0..=1.0).contains(&sample));
            }
        }
    }

    #[test]
    fn synthesized_effects_are_deterministic() {
        let a = synthesize_effects(44_100);
        let b = synthesize_effects(44_100);
        assert_eq!(a.place, b.place);
        assert_eq!(a.break_clip, b.break_clip);
        assert_eq!(a.footsteps[0], b.footsteps[0]);
        assert_eq!(a.footsteps[1], b.footsteps[1]);
        assert_eq!(a.footsteps[2], b.footsteps[2]);
        assert_eq!(a.jump, b.jump);
        assert_eq!(a.land, b.land);
    }
}
