extern crate clap;
extern crate rand;
extern crate rsrl;
extern crate rayon;
extern crate mm_arl;
extern crate csv;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use mm_arl::{
    TraderDomain,
    strategies::ExponentialUtilityStrategy,
    utils::Estimate,
};
use clap::{App, Arg};
use rayon::prelude::*;
use rsrl::domains::Domain;

#[derive(Debug, Serialize)]
struct Record {
    pub eta: f64,

    pub wealth_mean: f64,
    pub wealth_stddev: f64,

    pub inv_mean: f64,
    pub inv_stddev: f64,

    pub spread_mean: f64,
    pub spread_stddev: f64,
}

fn simulate(n_simulations: usize, eta: f64) -> Record {
    let mut pnls = vec![];
    let mut terminal_qs = vec![];
    let mut average_spread = vec![];

    for _ in 0..n_simulations {
        let mut domain = TraderDomain::default();
        let quotes = ExponentialUtilityStrategy::new(
            domain.dynamics.execution_dynamics.decay, eta,
            domain.dynamics.price_dynamics.volatility,
        );

        let mut a = quotes.compute(
            domain.dynamics.time,
            domain.dynamics.price,
            domain.inv,
        );

        let mut i = 1;
        let mut spread_sum = a[0] + a[1];

        loop {
            let t = domain.step(a);

            if t.terminated() {
                pnls.push(domain.wealth);
                terminal_qs.push(domain.inv_terminal);
                average_spread.push(spread_sum / i as f64);

                break
            } else {
                a = quotes.compute(
                    domain.dynamics.time,
                    domain.dynamics.price,
                    domain.inv,
                );

                i += 1;
                spread_sum += a[0] + a[1];
            }
        }
    }

    // Summarise results:
    let pnl_est = Estimate::from_slice(&pnls);
    let inv_est = Estimate::from_slice(&terminal_qs);
    let spd_est = Estimate::from_slice(&average_spread);

    Record {
        eta: eta,

        wealth_mean: pnl_est.0,
        wealth_stddev: pnl_est.1,

        inv_mean: inv_est.0,
        inv_stddev: inv_est.1,

        spread_mean: spd_est.0,
        spread_stddev: spd_est.1,
    }
}

fn main() {
    let matches = App::new("AS inventory strategy simulator")
        .arg(Arg::with_name("csv_path")
                .index(1)
                .required(true))
        .arg(Arg::with_name("n_simulations")
                .index(2)
                .required(true))
        .get_matches();

    let csv_path = matches.value_of("csv_path").unwrap();
    let n_simulations: usize = matches.value_of("n_simulations").unwrap().parse().unwrap();

    let mut records: Vec<_> = (1..101)
        .into_par_iter()
        .map(|i| 0.01 * i as f64)
        .chain(rayon::iter::once(0.001))
        .map(|g| simulate(n_simulations, g))
        .collect();
    records.par_sort_unstable_by(|a, b| a.eta.partial_cmp(&b.eta).unwrap());

    let mut file_logger = csv::Writer::from_path(csv_path).unwrap();

    for r in records {
        file_logger.serialize(r).ok();
    }

    file_logger.flush().ok();
}
