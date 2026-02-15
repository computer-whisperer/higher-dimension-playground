use rodio::{buffer::SamplesBuffer, Decoder, OutputStream, OutputStreamHandle, Sink};
use std::io::Cursor;
use std::sync::Arc;

const AUDIO_SAMPLE_RATE: u32 = 44_100;
const DEFAULT_MASTER_VOLUME: f32 = 1.0;
const EFFECT_BASE_GAINS: [f32; SoundEffect::COUNT] = [0.15, 0.18, 0.11, 0.14, 0.19];

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
        let idx = effect.index();
        let gain = (self.master_volume * EFFECT_BASE_GAINS[idx] * scale.max(0.0)).clamp(0.0, 4.0);
        if gain <= 0.0 {
            return;
        }

        let Ok(sink) = Sink::try_new(&output.handle) else {
            return;
        };
        sink.set_volume(gain);

        let mut rng = self.rng_state.get();
        let samples = self.effects[idx].pick(&mut rng).to_vec();
        self.rng_state.set(rng);

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
}
