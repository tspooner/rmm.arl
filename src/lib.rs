extern crate rand;
extern crate rand_distr;

extern crate rsrl;
extern crate slog;

pub mod utils;
pub mod dynamics;
pub mod strategies;

mod trader;
pub use self::trader::*;

mod adversary;
pub use self::adversary::*;

mod zero_sum;
pub use self::zero_sum::*;
