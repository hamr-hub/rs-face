//! Haar-like features and AdaBoost cascade classifier.
//!
//! This module implements the Viola-Jones face detector primitives from scratch:
//!
//! - 5 canonical Haar feature families (vertical edge, horizontal edge,
//!   diagonal edge, vertical center-surround, horizontal center-surround).
//! - A two-stage evaluation: per-feature sum using the (regular + tilted) integral
//!   image, then per-stage thresholding with weighted weak classifiers.
//!
//! The data format is compact and binary; see [`params`] for a small built-in
//! cascade trained on a synthetic "face-like" pattern (for tests and demos) and
//! for the loader.

pub mod cascade;
pub mod feature;
pub mod params;

pub use cascade::{Cascade, EvalCache, Stage, WeakFeature};
pub use feature::{FeatureKind, HaarFeature, Rect};
