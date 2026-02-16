use rodio::{buffer::SamplesBuffer, Decoder, OutputStream, OutputStreamHandle, Sink, SpatialSink};
use std::io::Cursor;
use std::sync::Arc;

const AUDIO_SAMPLE_RATE: u32 = 44_100;
const DEFAULT_MASTER_VOLUME: f32 = 1.0;
const EFFECT_BASE_GAINS: [f32; SoundEffect::COUNT] = [0.15, 0.18, 0.11, 0.14, 0.19];
const SPATIAL_EAR_DISTANCE: f32 = 0.18;
const SPATIAL_DISTANCE_SCALE: f32 = 0.35;
const SPATIAL_MIN_FORWARD_DEPTH: f32 = 0.02;

#[derive(Copy, Clone, Debug)]
pub enum SoundEffect {
    Place,
    Break,
    Footstep,
    Jump,
    Land,
}

pub type ViewBasis4 = ([f32; 4], [f32; 4], [f32; 4], [f32; 4]);

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

/// A set of sample variations for a single sound effect.
struct EffectVariations {
    samples: Vec<Arc<[f32]>>,
}

impl EffectVariations {
    fn single(s: Arc<[f32]>) -> Self {
        Self { samples: vec![s] }
    }

    fn pick(&self, rng: &mut u32) -> &Arc<[f32]> {
        let idx = if self.samples.len() > 1 {
            *rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*rng as usize) % self.samples.len()
        } else {
            0
        };
        &self.samples[idx]
    }
}

pub struct AudioEngine {
    output: Option<AudioOutput>,
    sample_rate: u32,
    pub master_volume: f32,
    effects: Vec<EffectVariations>,
    rng_state: std::cell::Cell<u32>,
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
            rng_state: std::cell::Cell::new(0x12345678),
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
        let Some((idx, gain)) = self.effect_gain(effect, scale) else {
            return;
        };

        let Ok(sink) = Sink::try_new(&output.handle) else {
            return;
        };
        sink.set_volume(gain);
        sink.append(self.pick_samples(idx));
        sink.detach();
    }

    pub fn play_spatial_scaled(
        &self,
        effect: SoundEffect,
        scale: f32,
        listener_position: [f32; 4],
        listener_basis: ViewBasis4,
        emitter_position: [f32; 4],
    ) {
        let Some(output) = &self.output else {
            return;
        };
        let Some((idx, gain)) = self.effect_gain(effect, scale) else {
            return;
        };

        let emitter =
            project_emitter_to_listener_space(listener_position, listener_basis, emitter_position);
        let half_ear = 0.5 * SPATIAL_EAR_DISTANCE;
        let left_ear = [-half_ear, 0.0, 0.0];
        let right_ear = [half_ear, 0.0, 0.0];

        let Ok(sink) = SpatialSink::try_new(&output.handle, emitter, left_ear, right_ear) else {
            self.play_scaled(effect, scale);
            return;
        };
        sink.set_volume(gain);
        sink.append(self.pick_samples(idx));
        sink.detach();
    }

    pub fn play_spatial_voxel_scaled(
        &self,
        effect: SoundEffect,
        scale: f32,
        listener_position: [f32; 4],
        listener_basis: ViewBasis4,
        voxel_position: [i32; 4],
    ) {
        self.play_spatial_scaled(
            effect,
            scale,
            listener_position,
            listener_basis,
            world_point_from_voxel_center(voxel_position),
        );
    }

    fn effect_gain(&self, effect: SoundEffect, scale: f32) -> Option<(usize, f32)> {
        let idx = effect.index();
        let gain = (self.master_volume * EFFECT_BASE_GAINS[idx] * scale.max(0.0)).clamp(0.0, 4.0);
        if gain <= 0.0 {
            return None;
        }
        Some((idx, gain))
    }

    fn pick_samples(&self, effect_idx: usize) -> SamplesBuffer<f32> {
        let mut rng = self.rng_state.get();
        let samples = self.effects[effect_idx].pick(&mut rng).to_vec();
        self.rng_state.set(rng);
        SamplesBuffer::new(1, self.sample_rate, samples)
    }
}

fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn project_emitter_to_listener_space(
    listener_position: [f32; 4],
    listener_basis: ViewBasis4,
    emitter_position: [f32; 4],
) -> [f32; 3] {
    let delta = [
        emitter_position[0] - listener_position[0],
        emitter_position[1] - listener_position[1],
        emitter_position[2] - listener_position[2],
        emitter_position[3] - listener_position[3],
    ];
    let (right, up, view_z, view_w) = listener_basis;
    let x = dot4(delta, right);
    let y = dot4(delta, up);
    let z = dot4(delta, view_z);
    let w = dot4(delta, view_w);

    // Match the 4D camera's ZW depth collapse so pan and attenuation track the viewport.
    let zw_depth = (z * z + w * w).sqrt().max(SPATIAL_MIN_FORWARD_DEPTH);
    [
        x * SPATIAL_DISTANCE_SCALE,
        y * SPATIAL_DISTANCE_SCALE,
        zw_depth * SPATIAL_DISTANCE_SCALE,
    ]
}

fn world_point_from_voxel_center(voxel_position: [i32; 4]) -> [f32; 4] {
    [
        voxel_position[0] as f32 + 0.5,
        voxel_position[1] as f32 + 0.5,
        voxel_position[2] as f32 + 0.5,
        voxel_position[3] as f32 + 0.5,
    ]
}

fn load_wav_samples(wav_bytes: &'static [u8], _sample_rate: u32) -> Arc<[f32]> {
    let cursor = Cursor::new(wav_bytes);
    let decoder = Decoder::new(cursor).expect("Failed to decode WAV");
    let samples: Vec<f32> = decoder.map(|s| s as f32 / i16::MAX as f32).collect();
    samples.into()
}

fn build_effect_table(sample_rate: u32) -> Vec<EffectVariations> {
    let load = |wav: &'static [u8]| load_wav_samples(wav, sample_rate);

    // Order must match SoundEffect::index()
    vec![
        // Place
        EffectVariations::single(load(include_bytes!(
            "../../../sfx5/place_block_01_medium.wav"
        ))),
        // Break
        EffectVariations::single(load(include_bytes!(
            "../../../sfx5/break_block_01_medium.wav"
        ))),
        // Footstep (3 variations)
        EffectVariations {
            samples: vec![
                load(include_bytes!("../../../sfx5/footstep_01_medium.wav")),
                load(include_bytes!("../../../sfx5/footstep_02_medium.wav")),
                load(include_bytes!("../../../sfx5/footstep_03_medium.wav")),
            ],
        },
        // Jump
        EffectVariations::single(load(include_bytes!("../../../sfx5/jump_01_medium.wav"))),
        // Land
        EffectVariations::single(load(include_bytes!("../../../sfx5/land_01_medium.wav"))),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_effects_are_non_empty() {
        let effects = build_effect_table(AUDIO_SAMPLE_RATE);
        assert_eq!(effects.len(), SoundEffect::COUNT);
        for effect in effects.iter() {
            for variation in &effect.samples {
                assert!(!variation.is_empty());
            }
        }
    }

    #[test]
    fn loaded_effects_are_normalized() {
        let effects = build_effect_table(AUDIO_SAMPLE_RATE);
        for effect in effects.iter() {
            for variation in &effect.samples {
                for sample in variation.iter() {
                    assert!(*sample >= -1.0);
                    assert!(*sample <= 1.0);
                }
            }
        }
    }

    #[test]
    fn projector_collapses_zw_depth() {
        let listener = [0.0, 0.0, 0.0, 0.0];
        let basis: ViewBasis4 = (
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        );

        let a = project_emitter_to_listener_space(listener, basis, [1.0, 0.0, 2.0, 2.0]);
        let b = project_emitter_to_listener_space(listener, basis, [1.0, 0.0, -2.0, -2.0]);

        assert!((a[0] - b[0]).abs() <= 1e-6);
        assert!((a[1] - b[1]).abs() <= 1e-6);
        assert!((a[2] - b[2]).abs() <= 1e-6);
    }
}
