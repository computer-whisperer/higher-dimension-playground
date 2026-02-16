use crate::audio_synth;
use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamHandle, Sink, SpatialSink};
use std::sync::Arc;

const AUDIO_SAMPLE_RATE: u32 = 44_100;
const DEFAULT_MASTER_VOLUME: f32 = 1.0;
const EFFECT_BASE_GAINS: [f32; SoundEffect::COUNT] = [0.15, 0.18, 0.11, 0.14, 0.19];
const SPATIAL_EAR_DISTANCE: f32 = 0.18;
const SPATIAL_DISTANCE_SCALE: f32 = 0.35;
const SPATIAL_MIN_FORWARD_DEPTH: f32 = 0.02;
const SPATIAL_MIN_DISTANCE: f32 = 1e-4;
pub const AUDIO_SPATIAL_FALLOFF_POWER_DEFAULT: f32 = 1.8;
pub const AUDIO_SPATIAL_FALLOFF_POWER_MIN: f32 = 1.0;
pub const AUDIO_SPATIAL_FALLOFF_POWER_MAX: f32 = 3.5;

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
    pub spatial_falloff_power: f32,
    effects: Vec<EffectVariations>,
    rng_state: std::cell::Cell<u32>,
}

impl AudioEngine {
    pub fn new(enabled: bool, master_volume: f32, spatial_falloff_power: f32) -> Self {
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
            spatial_falloff_power: spatial_falloff_power.clamp(
                AUDIO_SPATIAL_FALLOFF_POWER_MIN,
                AUDIO_SPATIAL_FALLOFF_POWER_MAX,
            ),
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

        let emitter = project_emitter_to_listener_space(
            listener_position,
            listener_basis,
            emitter_position,
            self.spatial_falloff_power,
        );
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
    falloff_power: f32,
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
    let base_distance = (x * x + y * y + zw_depth * zw_depth)
        .sqrt()
        .max(SPATIAL_MIN_DISTANCE);
    let distance_scaled = base_distance * SPATIAL_DISTANCE_SCALE;
    // Rodio applies ~1/dist^2 attenuation. Warp radial distance so we can target 1/r^N.
    let exponent = 0.5
        * falloff_power.clamp(
            AUDIO_SPATIAL_FALLOFF_POWER_MIN,
            AUDIO_SPATIAL_FALLOFF_POWER_MAX,
        );
    let warped_distance = distance_scaled.powf(exponent);
    let dir_scale = warped_distance / base_distance;
    [x * dir_scale, y * dir_scale, zw_depth * dir_scale]
}

fn world_point_from_voxel_center(voxel_position: [i32; 4]) -> [f32; 4] {
    [
        voxel_position[0] as f32 + 0.5,
        voxel_position[1] as f32 + 0.5,
        voxel_position[2] as f32 + 0.5,
        voxel_position[3] as f32 + 0.5,
    ]
}

fn build_effect_table(sample_rate: u32) -> Vec<EffectVariations> {
    let effects = audio_synth::synthesize_effects(sample_rate);

    // Order must match SoundEffect::index()
    vec![
        // Place
        EffectVariations::single(Arc::<[f32]>::from(effects.place)),
        // Break
        EffectVariations::single(Arc::<[f32]>::from(effects.break_clip)),
        // Footstep (3 variations)
        EffectVariations {
            samples: effects
                .footsteps
                .into_iter()
                .map(Arc::<[f32]>::from)
                .collect(),
        },
        // Jump
        EffectVariations::single(Arc::<[f32]>::from(effects.jump)),
        // Land
        EffectVariations::single(Arc::<[f32]>::from(effects.land)),
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

        let a = project_emitter_to_listener_space(
            listener,
            basis,
            [1.0, 0.0, 2.0, 2.0],
            AUDIO_SPATIAL_FALLOFF_POWER_DEFAULT,
        );
        let b = project_emitter_to_listener_space(
            listener,
            basis,
            [1.0, 0.0, -2.0, -2.0],
            AUDIO_SPATIAL_FALLOFF_POWER_DEFAULT,
        );

        assert!((a[0] - b[0]).abs() <= 1e-6);
        assert!((a[1] - b[1]).abs() <= 1e-6);
        assert!((a[2] - b[2]).abs() <= 1e-6);
    }

    #[test]
    fn projector_distance_warp_matches_falloff_power() {
        let listener = [0.0, 0.0, 0.0, 0.0];
        let basis: ViewBasis4 = (
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        );
        let emitter = [0.0, 0.0, 4.0, 0.0];

        let v2 = project_emitter_to_listener_space(listener, basis, emitter, 2.0);
        let v3 = project_emitter_to_listener_space(listener, basis, emitter, 3.0);
        let d2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
        let d3 = (v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2]).sqrt();

        assert!(d3 > d2);
    }
}
