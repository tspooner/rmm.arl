use crate::{
    dynamics::{ASDynamics, PoissonRate, BrownianMotionWithDrift},
    strategies::LinearUtilityTerminalPenaltyStrategy,
};
use rand::thread_rng;
use rsrl::{
    domains::{Domain, Transition, Observation},
    spaces::{
        real::Interval,
        ProductSpace,
    },
};

const MAX_DRIFT: f64 = 5.0;
const INV_BOUNDS: [f64; 2] = [-50.0, 50.0];

#[derive(Debug)]
pub struct AdversaryDomain<P, E> {
    pub dynamics: ASDynamics<P, E>,

    pub inv: f64,
    pub inv_terminal: f64,

    pub reward: f64,
    pub wealth: f64,

    inv_strategy: LinearUtilityTerminalPenaltyStrategy,
}

impl Default for AdversaryDomain<BrownianMotionWithDrift, PoissonRate> {
    fn default() -> Self {
        AdversaryDomain::default_with_eta(0.0)
    }
}

impl AdversaryDomain<BrownianMotionWithDrift, PoissonRate> {
    pub fn new(dynamics: ASDynamics<BrownianMotionWithDrift, PoissonRate>, eta: f64) -> Self {
        let inv_strategy = LinearUtilityTerminalPenaltyStrategy::new(
            dynamics.execution_dynamics.decay, eta,
        );

        Self {
            dynamics,

            inv: 0.0,
            inv_terminal: 0.0,

            reward: 0.0,
            wealth: 0.0,

            inv_strategy,
        }
    }

    pub fn default_with_eta(eta: f64) -> Self {
        let dynamics = ASDynamics::new(
            0.005, 100.0, thread_rng(),
            BrownianMotionWithDrift::new(0.005, 0.0, 2.0),
            PoissonRate::default()
        );

        Self::new(dynamics, eta)
    }

    fn do_executions(&mut self, ask_price: f64, bid_price: f64) {
        if self.inv > INV_BOUNDS[0] {
            if let Some(ask_offset) = self.dynamics.try_execute_ask(ask_price) {
                self.inv -= 1.0;
                self.reward -= ask_offset;
                self.wealth += ask_price;
            }
        }

        if self.inv < INV_BOUNDS[1] {
            if let Some(bid_offset) = self.dynamics.try_execute_bid(bid_price) {
                self.inv += 1.0;
                self.reward -= bid_offset;
                self.wealth -= bid_price;
            }
        }
    }

    fn update_state(&mut self, drift: f64) {
        let [ask_offset, bid_offset] = self.inv_strategy.compute(
            self.dynamics.time,
            self.dynamics.price,
            self.inv,
        );

        let ask_price = self.dynamics.price + ask_offset;
        let bid_price = self.dynamics.price - bid_offset;

        self.dynamics.price_dynamics.drift = MAX_DRIFT * (2.0 * drift - 1.0);
        self.reward = -(self.inv * self.dynamics.innovate());

        self.do_executions(ask_price, bid_price);

        if self.is_terminal() {
            // Execute market order favourably at midprice:
            self.wealth += self.dynamics.price * self.inv;

            self.inv_terminal = self.inv;
            self.inv = 0.0;
        }
    }

    fn is_terminal(&self) -> bool { self.dynamics.time >= 1.0 }
}

impl Domain for AdversaryDomain<BrownianMotionWithDrift, PoissonRate> {
    type StateSpace = ProductSpace<Interval>;
    type ActionSpace = Interval;

    fn emit(&self) -> Observation<Vec<f64>> {
        let state = vec![self.dynamics.time, self.inv.min(INV_BOUNDS[1]).max(INV_BOUNDS[0])];

        if self.is_terminal() {
            Observation::Terminal(state)
        } else {
            Observation::Full(state)
        }
    }

    fn step(&mut self, action: f64) -> Transition<Vec<f64>, f64> {
        let from = self.emit();
        let action = action.min(1.0).max(0.0);

        self.update_state(action);

        Transition {
            from,
            action,
            reward: self.reward,
            to: self.emit(),
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        ProductSpace::empty()
            + Interval::bounded(0.0, 1.0)
            + Interval::bounded(INV_BOUNDS[0], INV_BOUNDS[1])
    }

    fn action_space(&self) -> Interval {
        Interval::bounded(0.0, 1.0)
    }
}
