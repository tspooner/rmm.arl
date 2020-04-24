use crate::dynamics::{ASDynamics, PoissonRate, BrownianMotionWithDrift};
use rand::thread_rng;
use rsrl::{
    domains::{Domain, Transition, Observation},
    spaces::{
        real::{Reals, Interval},
        ProductSpace, TwoSpace, PairSpace,
    },
};

const INV_BOUNDS: [f64; 2] = [-50.0, 50.0];

pub struct ZeroSumDomain<P, E> {
    pub dynamics: ASDynamics<P, E>,

    pub inv: f64,
    pub inv_terminal: f64,

    pub reward: f64,
    pub wealth: f64,
}

impl Default for ZeroSumDomain<BrownianMotionWithDrift, PoissonRate> {
    fn default() -> Self {
        ZeroSumDomain::new(ASDynamics::new(
            0.005, 100.0, thread_rng(),
            BrownianMotionWithDrift::new(0.005, 0.0, 2.0),
            PoissonRate::default()
        ))
    }
}

impl ZeroSumDomain<BrownianMotionWithDrift, PoissonRate> {
    pub fn new(dynamics: ASDynamics<BrownianMotionWithDrift, PoissonRate>) -> Self {
        Self {
            dynamics,

            inv: 0.0,
            inv_terminal: 0.0,

            reward: 0.0,
            wealth: 0.0,
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

    fn update_state(&mut self, trader_action: [f64; 2], adversary_action: f64) {
        self.dynamics.price_dynamics.drift = adversary_action;
        self.reward = self.inv * self.dynamics.innovate();

        let ask_price = self.dynamics.price + trader_action[0];
        let bid_price = self.dynamics.price - trader_action[1];

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

impl Domain for ZeroSumDomain<BrownianMotionWithDrift, PoissonRate> {
    type StateSpace = ProductSpace<Interval>;
    type ActionSpace = PairSpace<TwoSpace<Reals>, Interval>;

    fn emit(&self) -> Observation<Vec<f64>> {
        let state = vec![self.dynamics.time, self.inv.min(INV_BOUNDS[1]).max(INV_BOUNDS[0])];

        if self.is_terminal() {
            Observation::Terminal(state)
        } else {
            Observation::Full(state)
        }
    }

    fn step(&mut self, action: ([f64; 2], f64)) -> Transition<Vec<f64>, ([f64; 2], f64)> {
        let from = self.emit();

        let trader_action = [
            action.0[0].max(0.0),
            action.0[1].max(0.0)
        ];
        let adversary_action = 10.0 * (2.0 * action.1 - 1.0);

        self.update_state(trader_action, adversary_action);

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

    fn action_space(&self) -> PairSpace<TwoSpace<Reals>, Interval> {
        PairSpace::new(TwoSpace::new([Reals; 2]), Interval::bounded(0.0, 1.0))
    }
}
