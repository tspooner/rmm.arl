extern crate mm_arl;
extern crate clap;
extern crate rand;
extern crate rsrl;
#[macro_use]
extern crate slog;
extern crate csv;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use mm_arl::{
    AdversaryDomain,
    utils::Estimate,
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
    policies::{Policy, Beta},
    prediction::{ValuePredictor, td::TD},
};

#[derive(Debug, Serialize)]
struct Record {
    pub wealth_mean: f64,
    pub wealth_stddev: f64,

    pub reward_mean: f64,
    pub reward_stddev: f64,

    pub inv_mean: f64,
    pub inv_stddev: f64,

    pub drift_mean: f64,
    pub drift_stddev: f64,

    pub value_estimate: f64,
    pub drift_neutral: f64,
    pub drift_bull: f64,
    pub drift_bear: f64,
}

fn main() {
    let matches = App::new("RL adversary")
        .arg(Arg::with_name("save_dir")
                .index(1)
                .required(true))
        .arg(Arg::with_name("eval_interval")
                .index(2)
                .required(true))
        .arg(Arg::with_name("eta")
                .long("eta")
                .required(false)
                .default_value("0.0"))
        .get_matches();

    let save_dir = matches.value_of("save_dir").unwrap();
    let eval_interval: usize = matches.value_of("eval_interval").unwrap().parse().unwrap();
    let eta: f64 = matches.value_of("eta").unwrap().parse().unwrap();

    let logger = logging::root(logging::stdout());
    let mut file_logger = csv::Writer::from_path(format!("{}/results.csv", save_dir)).unwrap();

    let mut rng = thread_rng();

    let domain_builder = || AdversaryDomain::default_with_eta(eta);

    // Build policy:
    let basis = Polynomial::new(2, 3).with_constant();

    let policy = Beta::new(
        TransformedLFA::scalar(basis.clone(), Softplus),
        TransformedLFA::scalar(basis.clone(), Softplus),
    );

    // Build critic:
    let critic = TD::new(LFA::scalar(basis, SGD(1.0)), 0.1, 1.0);

    // Build agent:
    let mut agent = TDAC::new(
        critic,
        policy,
        0.001,
        1.0,
    );

    // Pre-train value function:
    for _ in 0..1000 {
        let mut domain = domain_builder();
        let mut a = agent.sample_behaviour(&mut rng, domain.emit().state());

        loop {
            let t = domain.step(a);

            agent.critic.handle_transition(&t);

            if t.terminated() {
                break
            } else {
                a = agent.sample_behaviour(&mut rng, t.to.state());
            }
        }
    }

    for i in 0.. {
        let mut domain = domain_builder();
        let mut a = agent.sample_behaviour(&mut rng, domain.emit().state());

        loop {
            let t = domain.step(a);

            agent.handle_transition(&t);

            if t.terminated() {
                break
            } else {
                a = agent.sample_behaviour(&mut rng, t.to.state());
            }
        }

        OnlineLearner::<Vec<f64>, f64>::handle_terminal(&mut agent);

        if (i+1) % eval_interval == 0 {
            // Run an approximate evaluation:
            let mut pnls = vec![];
            let mut drifts = vec![];
            let mut rewards = vec![];
            let mut terminal_qs = vec![];

            for _ in 0..1000 {
                let mut domain = domain_builder();
                let mut a = agent.policy.mpa(domain.emit().state());

                use std::f64;

                let mut i = 0;
                let mut drift_sum = 0.0;
                let mut reward_sum = 0.0;

                loop {
                    let t = domain.step(a);

                    i += 1;
                    drift_sum += a;
                    reward_sum += t.reward;

                    if t.terminated() {
                        pnls.push(domain.wealth);
                        drifts.push(drift_sum / i as f64);
                        rewards.push(reward_sum);
                        terminal_qs.push(domain.inv_terminal);

                        break
                    } else {
                        a = agent.policy.mpa(t.to.state());
                    }
                }
            }

            // Summarise results:
            let pnl_est = Estimate::from_slice(&pnls);
            let rwd_est = Estimate::from_slice(&rewards);
            let inv_est = Estimate::from_slice(&terminal_qs);
            let dft_est = Estimate::from_slice(&drifts);

            // Log plotting data:
            let critic_est = agent.critic.predict_v(&vec![0.0, 0.0]);
            let drift_neutral = agent.policy.mpa(&vec![0.0, 0.0]);
            let drift_bull = agent.policy.mpa(&vec![0.0, 5.0]);
            let drift_bear = agent.policy.mpa(&vec![0.0, -5.0]);

            info!(logger, "evaluation {}", i / eval_interval;
                "wealth" => pnl_est,
                "reward" => rwd_est,
                "inv_terminal" => inv_est,
                "drift_mean" => Estimate::from_slice(&drifts),
                "critic" => critic_est,
                "drift_neutral" => drift_neutral,
                "drift_bull" => drift_bull,
                "drift_bear" => drift_bear,
            );

            file_logger.serialize(Record {
                wealth_mean: pnl_est.0,
                wealth_stddev: pnl_est.1,

                reward_mean: rwd_est.0,
                reward_stddev: rwd_est.1,

                inv_mean: inv_est.0,
                inv_stddev: inv_est.1,

                drift_mean: dft_est.0,
                drift_stddev: dft_est.1,

                value_estimate: agent.critic.predict_v(&vec![0.0, 0.0]),
                drift_neutral: agent.policy.mpa(&vec![0.0, 0.0]),
                drift_bull: agent.policy.mpa(&vec![0.0, 5.0]),
                drift_bear: agent.policy.mpa(&vec![0.0, -5.0]),
            }).ok();
            file_logger.flush().ok();
        }
    }
}
