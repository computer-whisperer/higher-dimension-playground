use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamHandle, Sink};
use std::f32::consts::PI;
use std::sync::Arc;

const AUDIO_SAMPLE_RATE: u32 = 44_100;
const DEFAULT_MASTER_VOLUME: f32 = 0.7;
const EFFECT_BASE_GAINS: [f32; SoundEffect::COUNT] = [0.75, 0.90, 0.55, 0.70, 0.95];

#[derive(Copy, Clone, Debug)]
pub enum SoundEffect {
    Place,
    Break,
    Footstep,
    Jump,
    Land,
}

impl SoundEffect {
    const COUNT: usize = 5;

    fn index(self) -> usize {
        match self {
            Self::Place => 0,
            Self::Break => 1,
            Self::Footstep => 2,
            Self::Jump => 3,
            Self::Land => 4,
        }
    }
}

struct AudioOutput {
    _stream: OutputStream,
    handle: OutputStreamHandle,
}

pub struct AudioEngine {
    output: Option<AudioOutput>,
    sample_rate: u32,
    master_volume: f32,
    effects: [Arc<[f32]>; SoundEffect::COUNT],
}

impl AudioEngine {
    pub fn new(enabled: bool, master_volume: f32) -> Self {
        let sample_rate = AUDIO_SAMPLE_RATE;
        let effects = build_effect_table(sample_rate);
        let output = if enabled {
            match OutputStream::try_default() {
                Ok((stream, handle)) => Some(AudioOutput {
                    _stream: stream,
                    handle,
                }),
                Err(error) => {
                    eprintln!("Audio output unavailable: {error}");
                    None
                }
            }
        } else {
            None
        };

        Self {
            output,
            sample_rate,
            master_volume: master_volume.clamp(0.0, 2.0).max(if enabled {
                0.0
            } else {
                DEFAULT_MASTER_VOLUME
            }),
            effects,
        }
    }

    pub fn is_active(&self) -> bool {
        self.output.is_some()
    }

    pub fn play(&self, effect: SoundEffect) {
        self.play_scaled(effect, 1.0);
    }

    pub fn play_scaled(&self, effect: SoundEffect, scale: f32) {
        let Some(output) = &self.output else {
            return;
        };
        let idx = effect.index();
        let gain = (self.master_volume * EFFECT_BASE_GAINS[idx] * scale.max(0.0)).clamp(0.0, 4.0);
        if gain <= 0.0 {
            return;
        }

        let Ok(sink) = Sink::try_new(&output.handle) else {
            return;
        };
        sink.set_volume(gain);
        sink.append(SamplesBuffer::new(
            1,
            self.sample_rate,
            self.effects[idx].to_vec(),
        ));
        sink.detach();
    }
}

fn build_effect_table(sample_rate: u32) -> [Arc<[f32]>; SoundEffect::COUNT] {
    [
        Arc::<[f32]>::from(synth_place(sample_rate)),
        Arc::<[f32]>::from(synth_break(sample_rate)),
        Arc::<[f32]>::from(synth_footstep(sample_rate)),
        Arc::<[f32]>::from(synth_jump(sample_rate)),
        Arc::<[f32]>::from(synth_land(sample_rate)),
    ]
}

fn next_noise(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let v = ((*seed >> 9) & 0x007f_ffff) as f32 / 8_388_607.0;
    v * 2.0 - 1.0
}

fn clamp_sample(sample: f32) -> f32 {
    sample.clamp(-1.0, 1.0)
}

fn synth_place(sample_rate: u32) -> Vec<f32> {
    let duration_s = 0.09;
    let count = (duration_s * sample_rate as f32) as usize;
    let mut out = Vec::with_capacity(count);
    let mut seed = 0x1234_5678;
    for i in 0..count {
        let t = i as f32 / sample_rate as f32;
        let env = (-34.0 * t).exp();
        let tone = (2.0 * PI * (820.0 * t - 190.0 * t * t)).sin();
        let sparkle = next_noise(&mut seed) * 0.08;
        out.push(clamp_sample((tone * 0.78 + sparkle) * env));
    }
    out
}

fn synth_break(sample_rate: u32) -> Vec<f32> {
    let duration_s = 0.18;
    let count = (duration_s * sample_rate as f32) as usize;
    let mut out = Vec::with_capacity(count);
    let mut seed = 0x9e37_79b9;
    for i in 0..count {
        let t = i as f32 / sample_rate as f32;
        let env = (-20.0 * t).exp();
        let noise = next_noise(&mut seed);
        let rumble = (2.0 * PI * 92.0 * t).sin() * (-10.0 * t).exp();
        let grit = noise.signum() * noise * noise.abs();
        out.push(clamp_sample((grit * 0.70 + rumble * 0.45) * env));
    }
    out
}

fn synth_footstep(sample_rate: u32) -> Vec<f32> {
    let duration_s = 0.12;
    let count = (duration_s * sample_rate as f32) as usize;
    let mut out = Vec::with_capacity(count);
    let mut seed = 0xa24b_aed4;
    for i in 0..count {
        let t = i as f32 / sample_rate as f32;
        let env = (-26.0 * t).exp();
        let thump = (2.0 * PI * (105.0 - 45.0 * t) * t).sin() * 0.70;
        let grit = next_noise(&mut seed) * 0.30;
        out.push(clamp_sample((thump + grit) * env));
    }
    out
}

fn synth_jump(sample_rate: u32) -> Vec<f32> {
    let duration_s = 0.14;
    let count = (duration_s * sample_rate as f32) as usize;
    let mut out = Vec::with_capacity(count);
    let mut seed = 0x7f4a_7c15;
    for i in 0..count {
        let t = i as f32 / sample_rate as f32;
        let env = (-18.0 * t).exp();
        let chirp = (2.0 * PI * (360.0 * t + 1020.0 * t * t)).sin();
        let airy = next_noise(&mut seed) * 0.09;
        out.push(clamp_sample((chirp * 0.62 + airy) * env));
    }
    out
}

fn synth_land(sample_rate: u32) -> Vec<f32> {
    let duration_s = 0.20;
    let count = (duration_s * sample_rate as f32) as usize;
    let mut out = Vec::with_capacity(count);
    let mut seed = 0xbf58_476d;
    for i in 0..count {
        let t = i as f32 / sample_rate as f32;
        let env = (-22.0 * t).exp();
        let impact = (2.0 * PI * (74.0 - 20.0 * t) * t).sin() * (-8.0 * t).exp();
        let dust = next_noise(&mut seed) * 0.33;
        out.push(clamp_sample((impact * 0.95 + dust) * env));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_effects_are_non_empty() {
        let effects = build_effect_table(AUDIO_SAMPLE_RATE);
        for effect in effects {
            assert!(!effect.is_empty());
            assert!(effect.len() > 500);
        }
    }

    #[test]
    fn generated_effects_are_clamped() {
        let effects = build_effect_table(AUDIO_SAMPLE_RATE);
        for effect in effects {
            for sample in effect.iter() {
                assert!(*sample >= -1.0);
                assert!(*sample <= 1.0);
            }
        }
    }
}
