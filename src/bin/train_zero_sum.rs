extern crate mm_arl;
extern crate clap;
extern crate rand;
extern crate rsrl;
#[macro_use]
extern crate slog;
extern crate csv;
extern crate serde;

use mm_arl::{
    ZeroSumDomain,
    utils::Estimate
};
use clap::{App, Arg};
use rand::thread_rng;
use rsrl::{
    OnlineLearner,
    control::{Controller, ac::TDAC},
    domains::Domain,
    fa::{
        TransformedLFA,
        linear::{LFA, basis::{Projector, Polynomial}, optim::SGD},
        transforms::Softplus,
    },
    logging,
    policies::{Policy, Beta, IPP, gaussian::{self, Gaussian}},
    prediction::{ValuePredictor, td::TD},
};
use std::fs::File;

fn main() {
    let matches = App::new("ZS training")
        .arg(Arg::with_name("eval_interval")
                .index(1)
                .required(true))
        .get_matches();

    let eval_interval: usize = matches.value_of("eval_interval").unwrap().parse().unwrap();

    let logger = logging::root(logging::stdout());
    let file_logger = logging::root(logging::file(
        File::create(format!("/tmp/performance.txt")).expect("Failed to create log file.")
    ));

    let mut rng = thread_rng();
    let mut trader = {
        let basis = Polynomial::new(2, 3).with_constant();

        // Build policy:
        let policy_a = Gaussian::new(
            gaussian::mean::Scalar(LFA::scalar(basis.clone(), SGD(1.0))),
            gaussian::stddev::Scalar(TransformedLFA::scalar(basis.clone(), Softplus)),
        );
        let policy_b = Gaussian::new(
            gaussian::mean::Scalar(TransformedLFA::scalar(basis.clone(), Softplus)),
            gaussian::stddev::Scalar(TransformedLFA::scalar(basis.clone(), Softplus)),
        );
        let policy = IPP::new(policy_a, policy_b);

        // Build critic:
        let critic = TD::new(LFA::scalar(basis.clone(), SGD(1.0)), 0.01, 1.0);

        // Build agent:
        TDAC::new(
            critic,
            policy,
            0.00001,
            1.0,
        )
    };

    let mut adversary = {
        let basis = Polynomial::new(2, 5).with_constant();

        // Build policy:
        let policy = Beta::new(
            TransformedLFA::scalar(basis.clone(), Softplus),
            TransformedLFA::scalar(basis.clone(), Softplus),
        );

        // Build critic:
        let critic = TD::new(LFA::scalar(basis, SGD(1.0)), 0.01, 1.0);

        // Build agent:
        TDAC::new(
            critic,
            policy,
            0.1,
            1.0,
        )
    };

    fn ua_(a: (f64, f64)) -> [f64; 2] {
        [
            a.0 + a.1,
            a.1 - a.0
        ]
    }

    // Pre-train value function:
    for _ in 0..1000 {
        let mut domain = ZeroSumDomain::default();
        let mut a = (
            ua_(trader.sample_behaviour(&mut rng, domain.emit().state())),
            adversary.sample_behaviour(&mut rng, domain.emit().state())
        );

        loop {
            let t = domain.step(a);

            trader.critic.handle_transition(&t);
            adversary.critic.handle_transition(&t);

            if t.terminated() {
                break
            } else {
                a = (
                    ua_(trader.sample_behaviour(&mut rng, domain.emit().state())),
                    adversary.sample_behaviour(&mut rng, domain.emit().state())
                );
            }
        }
    }

    for i in 0.. {
        let mut domain = ZeroSumDomain::default();
        let mut a = (
            ua_(trader.sample_behaviour(&mut rng, domain.emit().state())),
            adversary.sample_behaviour(&mut rng, domain.emit().state())
        );

        loop {
            let t = domain.step(a);
            let is_terminal = t.terminated();

            trader.handle_transition(&t.clone().replace_action((a.0[0], a.0[1])));
            adversary.handle_transition(&t.replace_action(a.1).negate_reward());

            if is_terminal {
                break
            } else {
                a = (
                    ua_(trader.sample_behaviour(&mut rng, domain.emit().state())),
                    adversary.sample_behaviour(&mut rng, domain.emit().state())
                );
            }
        }

        OnlineLearner::<Vec<f64>, (f64, f64)>::handle_terminal(&mut trader);
        OnlineLearner::<Vec<f64>, f64>::handle_terminal(&mut adversary);

        if (i+1) % eval_interval == 0 {
            // Run an approximate evaluation:
            let mut pnls = vec![];
            let mut rewards = vec![];
            let mut terminal_qs = vec![];
            let mut average_spread = vec![];

            for _ in 0..1000 {
                let mut domain = ZeroSumDomain::default();
                let mut a = (
                    ua_(trader.policy.mpa(domain.emit().state())),
                    adversary.policy.mpa(domain.emit().state())
                );

                let mut i = 1;
                let mut reward_sum = 0.0;
                let mut spread_sum = a.0[0] + a.0[1];

                loop {
                    let t = domain.step(a);

                    reward_sum += t.reward;

                    if t.terminated() {
                        pnls.push(domain.wealth);
                        rewards.push(reward_sum);
                        terminal_qs.push(domain.inv_terminal);
                        average_spread.push(spread_sum / i as f64);

                        break
                    } else {
                        a = (
                            ua_(trader.policy.mpa(domain.emit().state())),
                            adversary.policy.mpa(domain.emit().state())
                        );

                        i += 1;
                        spread_sum += a.0[0] + a.0[1];
                    }
                }
            }

            let pnl_est = Estimate::from_slice(&pnls);
            let reward_est = Estimate::from_slice(&rewards);

            // Log to stdout:
            info!(logger, "evaluation {}", i / eval_interval;
                "wealth" => pnl_est,
                "reward" => reward_est,
                "critic" => trader.critic.predict_v(&vec![0.0, 0.0]),
                "inv_terminal" => Estimate::from_slice(&terminal_qs),
                "spread" => Estimate::from_slice(&average_spread),
            );

            let performance = Estimate::from_slice(&pnls);
            info!(file_logger, "{},{}", performance.0, performance.1);

            let d_logger = logging::root(logging::file(
                File::create("/tmp/returns.txt").expect("Failed to create log file.")
            ));
            for x in pnls.iter() { info!(d_logger, "{}", x); }
        }
    }
}
