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
    TraderDomain,
    dynamics::ASDynamics,
    utils::Estimate,
};
use clap::{App, Arg};
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
    policies::{Policy, IPP, gaussian::{self, Gaussian}},
    prediction::{ValuePredictor, td::TD},
};
use std::f64;

fn mean(x: [f64; 2]) -> f64 { (x[0] - x[1]) / 2.0 }

#[derive(Debug, Serialize)]
struct Record {
    pub episode: usize,

    pub wealth_mean: f64,
    pub wealth_stddev: f64,

    pub reward_mean: f64,
    pub reward_stddev: f64,

    pub inv_mean: f64,
    pub inv_stddev: f64,

    pub spread_mean: f64,
    pub spread_stddev: f64,

    pub value_estimate: f64,
    pub rp_neutral: f64,
    pub rp_bull: f64,
    pub rp_bear: f64,
}

fn main() {
    let matches = App::new("RL trader")
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

    let mut rng = rand::thread_rng();

    let domain_builder = || TraderDomain::new(ASDynamics::default(), eta);

    // Build basis:
    let basis = Polynomial::new(2, 3).with_constant();

    // Build policy:
    let policy_rp = Gaussian::new(
        gaussian::mean::Scalar(LFA::scalar(basis.clone(), SGD(1.0))),
        gaussian::stddev::Scalar(TransformedLFA::scalar(basis.clone(), Softplus)),
    );
    let policy_sp = Gaussian::new(
        gaussian::mean::Scalar(TransformedLFA::scalar(basis.clone(), Softplus)),
        gaussian::stddev::Scalar(TransformedLFA::scalar(basis.clone(), Softplus)),
    );
    let policy = IPP::new(policy_rp, policy_sp);

    // Build critic:
    let critic = TD::new(
        LFA::scalar(basis.clone(), SGD(1.0)),
        0.01,
        1.0
    );

    // Build agent:
    let mut agent = TDAC::new(
        critic,
        policy,
        0.000001,
        1.0,
    );

    fn ua_(a: (f64, f64)) -> [f64; 2] {
        [
            a.0 + a.1,
            a.1 - a.0
        ]
    }

    // Pre-train value function:
    for _ in 0..1000 {
        let mut domain = domain_builder();
        let mut a = agent.sample_behaviour(&mut rng, domain.emit().state());

        loop {
            let a_ = ua_(a);
            let t = domain.step(a_);

            agent.critic.handle_transition(&t);

            if t.terminated() {
                break
            } else {
                a = agent.sample_behaviour(&mut rng, t.to.state());
            }
        }
    }

    // Run experiment:
    for i in 0..(1000*eval_interval) {
        // Perform evaluation:
        if i % eval_interval == 0 {
            let mut pnls = vec![];
            let mut rewards = vec![];
            let mut terminal_qs = vec![];
            let mut average_spread = vec![];

            for _ in 0..1000 {
                let mut domain = domain_builder();
                let mut a = agent.sample_target(&mut rng, domain.emit().state());

                let mut i = 1;
                let mut reward_sum = 0.0;
                let mut spread_sum = a.1 * 2.0;

                loop {
                    let a_ = ua_(a);
                    let t = domain.step(a_);

                    reward_sum += t.reward;

                    if t.terminated() {
                        pnls.push(domain.wealth);
                        rewards.push(reward_sum);
                        terminal_qs.push(domain.inv_terminal);
                        average_spread.push(spread_sum / i as f64);

                        break
                    } else {
                        a = agent.sample_target(&mut rng, t.to.state());

                        i += 1;
                        spread_sum += a.1 * 2.0;
                    }
                }
            }

            // Summarise results:
            let pnl_est = Estimate::from_slice(&pnls);
            let rwd_est = Estimate::from_slice(&rewards);
            let inv_est = Estimate::from_slice(&terminal_qs);
            let spd_est = Estimate::from_slice(&average_spread);

            // Log plotting data:
            let critic_est = agent.critic.predict_v(&vec![0.0, 0.0]);
            let rp_neutral = mean(ua_(agent.policy.mpa(&vec![0.0, 0.0])));
            let rp_bull = mean(ua_(agent.policy.mpa(&vec![0.0, 5.0])));
            let rp_bear = mean(ua_(agent.policy.mpa(&vec![0.0, -5.0])));

            info!(logger, "evaluation {}", i / eval_interval;
                "wealth" => pnl_est,
                "reward" => rwd_est,
                "inv_terminal" => inv_est,
                "average_spread" => spd_est,
                "critic" => critic_est,
                "rp_neutral" => rp_neutral,
                "rp_bull" => rp_bull,
                "rp_bear" => rp_bear,
            );

            file_logger.serialize(Record {
                episode: i,

                wealth_mean: pnl_est.0,
                wealth_stddev: pnl_est.1,

                reward_mean: rwd_est.0,
                reward_stddev: rwd_est.1,

                inv_mean: inv_est.0,
                inv_stddev: inv_est.1,

                spread_mean: spd_est.0,
                spread_stddev: spd_est.1,

                value_estimate: agent.critic.predict_v(&vec![0.0, 0.0]),
                rp_neutral: rp_neutral,
                rp_bull: rp_bull,
                rp_bear: rp_bear,
            }).ok();
            file_logger.flush().ok();
        }

        // Train agent for one episode:
        let mut domain = domain_builder();
        let mut a = agent.sample_behaviour(&mut rng, domain.emit().state());

        loop {
            let a_ = ua_(a);
            let t = domain.step(a_).replace_action(a);

            agent.handle_transition(&t);

            if t.terminated() {
                break
            } else {
                a = agent.sample_behaviour(&mut rng, t.to.state());
            }
        }

        OnlineLearner::<Vec<f64>, (f64, f64)>::handle_terminal(&mut agent);
    }
}
