use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Build a deterministic RNG. If `seed` is `None`, a non-deterministic seed
/// is drawn from the OS.
pub fn make_rng(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}
