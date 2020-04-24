use crate::dynamics::{ASDynamics, PriceDynamics, ExecutionDynamics, PoissonRate, BrownianMotion};
use rand::thread_rng;
use rsrl::{
    domains::{Domain, Transition, Observation},
    spaces::{
        real::{Reals, Interval},
        ProductSpace, TwoSpace,
    },
};

const INV_BOUNDS: [f64; 2] = [-50.0, 50.0];

#[derive(Debug)]
pub struct TraderDomain<P, E> {
    pub dynamics: ASDynamics<P, E>,

    pub inv: f64,
    pub inv_terminal: f64,

    pub reward: f64,
    pub wealth: f64,

    eta: f64,
}

impl Default for TraderDomain<BrownianMotion, PoissonRate> {
    fn default() -> Self {
        TraderDomain::new(ASDynamics::default(), 0.0)
    }
}

impl TraderDomain<BrownianMotion, PoissonRate> {
    pub fn default_with_eta(eta: f64) -> Self {
        let dynamics = ASDynamics::new(
            0.005, 100.0, thread_rng(),
            BrownianMotion::new(0.005, 2.0),
            PoissonRate::default()
        );

        Self::new(dynamics, eta)
    }
}

impl<P, E> TraderDomain<P, E>
where
    P: PriceDynamics,
    E: ExecutionDynamics,
{
    pub fn new(dynamics: ASDynamics<P, E>, eta: f64) -> Self {
        Self {
            dynamics,

            inv: 0.0,
            inv_terminal: 0.0,

            reward: 0.0,
            wealth: 0.0,

            eta,
        }
    }

    fn do_executions(&mut self, ask_price: f64, bid_price: f64) {
        if self.inv > INV_BOUNDS[0] {
            if let Some(ask_offset) = self.dynamics.try_execute_ask(ask_price) {
                self.inv -= 1.0;
                self.reward += ask_offset;
                self.wealth += ask_price;
            }
        }

        if self.inv < INV_BOUNDS[1] {
            if let Some(bid_offset) = self.dynamics.try_execute_bid(bid_price) {
                self.inv += 1.0;
                self.reward += bid_offset;
                self.wealth -= bid_price;
            }
        }
    }

    fn update_state(&mut self, ask_offset: f64, bid_offset: f64) {
        let ask_price = self.dynamics.price + ask_offset;
        let bid_price = self.dynamics.price - bid_offset;

        self.reward = self.inv * self.dynamics.innovate();

        self.do_executions(ask_price, bid_price);

        if self.is_terminal() {
            // Execute market order favourably at midprice:
            self.wealth += self.dynamics.price * self.inv;
            self.reward -= self.eta * self.inv.powi(2);

            self.inv_terminal = self.inv;
            self.inv = 0.0;
        }
    }

    fn is_terminal(&self) -> bool { self.dynamics.time >= 1.0 }
}

impl<P, E> Domain for TraderDomain<P, E>
where
    P: PriceDynamics,
    E: ExecutionDynamics,
{
    type StateSpace = ProductSpace<Interval>;
    type ActionSpace = TwoSpace<Reals>;

    fn emit(&self) -> Observation<Vec<f64>> {
        let state = vec![self.dynamics.time, self.inv.min(INV_BOUNDS[1]).max(INV_BOUNDS[0])];

        if self.is_terminal() {
            Observation::Terminal(state)
        } else {
            Observation::Full(state)
        }
    }

    fn step(&mut self, action: [f64; 2]) -> Transition<Vec<f64>, [f64; 2]> {
        let from = self.emit();

        self.update_state(action[0], action[1]);

        Transition {
            from,
            action,
            to: self.emit(),
            reward: self.reward// - self.eta * self.inv.powi(2),
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        ProductSpace::empty()
            + Interval::bounded(0.0, 1.0)
            + Interval::bounded(INV_BOUNDS[0], INV_BOUNDS[1])
    }

    fn action_space(&self) -> TwoSpace<Reals> {
        TwoSpace::new([Reals; 2])
    }
}
