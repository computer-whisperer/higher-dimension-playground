use crate::shared::region_tree::RegionTreeCore;
use crate::shared::spatial::Aabb4i;
use std::sync::Arc;

mod legacy_generator;

pub use legacy_generator::LegacyWorldGenerator;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueryDetail {
    Coarse,
    Exact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryVolume {
    pub bounds: Aabb4i,
}

pub trait WorldField {
    fn query_region_core(&self, query: QueryVolume, detail: QueryDetail) -> Arc<RegionTreeCore>;
}

pub trait WorldOverlay: WorldField {}

#[derive(Debug)]
pub struct PassthroughWorldOverlay<F> {
    field: F,
}

impl<F> PassthroughWorldOverlay<F> {
    pub fn new(field: F) -> Self {
        Self { field }
    }

    pub fn field(&self) -> &F {
        &self.field
    }

    pub fn field_mut(&mut self) -> &mut F {
        &mut self.field
    }

    pub fn into_inner(self) -> F {
        self.field
    }
}

impl<F> WorldField for PassthroughWorldOverlay<F>
where
    F: WorldField,
{
    fn query_region_core(&self, query: QueryVolume, detail: QueryDetail) -> Arc<RegionTreeCore> {
        self.field.query_region_core(query, detail)
    }
}

impl<F> WorldOverlay for PassthroughWorldOverlay<F> where F: WorldField {}

pub type ServerWorldField = LegacyWorldGenerator;
pub type ServerWorldOverlay = PassthroughWorldOverlay<LegacyWorldGenerator>;
