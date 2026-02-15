use rodio::{buffer::SamplesBuffer, Decoder, OutputStream, OutputStreamHandle, Sink};
use std::io::Cursor;
use std::sync::Arc;

const AUDIO_SAMPLE_RATE: u32 = 44_100;
const DEFAULT_MASTER_VOLUME: f32 = 1.0;
const EFFECT_BASE_GAINS: [f32; SoundEffect::COUNT] = [0.15, 0.18, 0.11, 0.14, 0.19, 0.13, 0.12, 0.10];
const FOOTSTEP_VARIATION_COUNT: usize = 3;

#[derive(Copy, Clone, Debug)]
pub enum SoundEffect {
    Place,
    Break,
    Footstep,
    Jump,
    Land,
    Hurt,
    Pickup,
    UiTick,
}

impl SoundEffect {
    const COUNT: usize = 8;

    fn index(self) -> usize {
        match self {
            Self::Place => 0,
            Self::Break => 1,
            Self::Footstep => 2,
            Self::Jump => 3,
            Self::Land => 4,
            Self::Hurt => 5,
            Self::Pickup => 6,
            Self::UiTick => 7,
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
    pub master_volume: f32,
    effects: [Arc<[f32]>; SoundEffect::COUNT],
    footstep_variations: [Arc<[f32]>; FOOTSTEP_VARIATION_COUNT],
    footstep_rng_state: std::cell::Cell<u32>,
}

impl AudioEngine {
    pub fn new(enabled: bool, master_volume: f32) -> Self {
        let sample_rate = AUDIO_SAMPLE_RATE;
        let (effects, footstep_variations) = build_effect_table(sample_rate);
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
            footstep_variations,
            footstep_rng_state: std::cell::Cell::new(0x12345678),
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

        // For footsteps, randomly select one of the variations
        let samples = if matches!(effect, SoundEffect::Footstep) {
            let mut rng = self.footstep_rng_state.get();
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.footstep_rng_state.set(rng);
            let variant = (rng % FOOTSTEP_VARIATION_COUNT as u32) as usize;
            self.footstep_variations[variant].to_vec()
        } else {
            self.effects[idx].to_vec()
        };

        sink.append(SamplesBuffer::new(1, self.sample_rate, samples));
        sink.detach();
    }
}

fn load_wav_samples(wav_bytes: &'static [u8], _sample_rate: u32) -> Arc<[f32]> {
    let cursor = Cursor::new(wav_bytes);
    let decoder = Decoder::new(cursor).expect("Failed to decode WAV");
    let samples: Vec<f32> = decoder.map(|s| s as f32 / i16::MAX as f32).collect();
    samples.into()
}

fn build_effect_table(sample_rate: u32) -> ([Arc<[f32]>; SoundEffect::COUNT], [Arc<[f32]>; FOOTSTEP_VARIATION_COUNT]) {
    // Embed WAV files at compile time
    const PLACE_WAV: &[u8] = include_bytes!("../../../sfx/place_block_01.wav");
    const BREAK_WAV: &[u8] = include_bytes!("../../../sfx/break_block_01.wav");
    const FOOTSTEP_01_WAV: &[u8] = include_bytes!("../../../sfx/footstep_01.wav");
    const FOOTSTEP_02_WAV: &[u8] = include_bytes!("../../../sfx/footstep_02.wav");
    const FOOTSTEP_03_WAV: &[u8] = include_bytes!("../../../sfx/footstep_03.wav");
    const JUMP_WAV: &[u8] = include_bytes!("../../../sfx/jump_01.wav");
    const LAND_WAV: &[u8] = include_bytes!("../../../sfx/land_01.wav");
    const HURT_WAV: &[u8] = include_bytes!("../../../sfx/hurt_01.wav");
    const PICKUP_WAV: &[u8] = include_bytes!("../../../sfx/pickup_01.wav");
    const UI_TICK_WAV: &[u8] = include_bytes!("../../../sfx/ui_tick_01.wav");

    let effects = [
        load_wav_samples(PLACE_WAV, sample_rate),
        load_wav_samples(BREAK_WAV, sample_rate),
        load_wav_samples(FOOTSTEP_01_WAV, sample_rate), // Default footstep (not used in play_scaled)
        load_wav_samples(JUMP_WAV, sample_rate),
        load_wav_samples(LAND_WAV, sample_rate),
        load_wav_samples(HURT_WAV, sample_rate),
        load_wav_samples(PICKUP_WAV, sample_rate),
        load_wav_samples(UI_TICK_WAV, sample_rate),
    ];

    let footstep_variations = [
        load_wav_samples(FOOTSTEP_01_WAV, sample_rate),
        load_wav_samples(FOOTSTEP_02_WAV, sample_rate),
        load_wav_samples(FOOTSTEP_03_WAV, sample_rate),
    ];

    (effects, footstep_variations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_effects_are_non_empty() {
        let (effects, footstep_variations) = build_effect_table(AUDIO_SAMPLE_RATE);
        for effect in effects.iter() {
            assert!(!effect.is_empty());
        }
        for footstep in footstep_variations.iter() {
            assert!(!footstep.is_empty());
        }
    }

    #[test]
    fn loaded_effects_are_normalized() {
        let (effects, footstep_variations) = build_effect_table(AUDIO_SAMPLE_RATE);
        for effect in effects.iter() {
            for sample in effect.iter() {
                assert!(*sample >= -1.0);
                assert!(*sample <= 1.0);
            }
        }
        for footstep in footstep_variations.iter() {
            for sample in footstep.iter() {
                assert!(*sample >= -1.0);
                assert!(*sample <= 1.0);
            }
        }
    }
}
