extern crate clap;
extern crate rand;
extern crate rsrl;
extern crate rayon;
extern crate mm_arl;

use mm_arl::{
    TraderDomain,
    dynamics::ASDynamics,
    strategies::LinearUtilityStrategy,
    utils::{mean_var, median_quantiles},
};
use clap::{App, Arg};
use rayon::prelude::*;
use rsrl::domains::Domain;

fn simulate_once(risk_param: f64) -> (f64, f64) {
    let mut domain = TraderDomain::new(ASDynamics::default_with_drift(0.0), risk_param);

    let quotes = LinearUtilityStrategy::new(
        domain.dynamics.execution_dynamics.decay,
    );

    loop {
        let a = quotes.compute(
            domain.dynamics.time,
            domain.dynamics.price,
            domain.inv,
        );
        let t = domain.step(a);

        if t.terminated() {
            return (domain.wealth, domain.inv_terminal)
        }
    }
}

fn main() {
    let matches = App::new("AS inventory strategy simulator")
        .arg(Arg::with_name("n_simulations")
                .index(1)
                .required(true))
        .arg(Arg::with_name("risk_param")
                .index(2)
                .required(true))
        .get_matches();

    let n_simulations: usize = matches.value_of("n_simulations").unwrap().parse().unwrap();
    let risk_param: f64 = matches.value_of("risk_param").unwrap().parse().unwrap();

    let (mut pnls, mut terminal_qs): (Vec<_>, Vec<_>) =
        (0..n_simulations).into_par_iter().map(move |_| simulate_once(risk_param)).unzip();

    pnls.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    terminal_qs.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let [mean, var] = mean_var(&pnls);
    let [q25, median, q75] = median_quantiles(&pnls);
    println!("PnL: {} pm {} | {} < {} < {}", mean, var.sqrt(), q25, median, q75);

    let [mean, var] = mean_var(&terminal_qs);
    let [q25, median, q75] = median_quantiles(&terminal_qs);
    println!("Inv: {} pm {} | {} < {} < {}", mean, var.sqrt(), q25, median, q75);
}
