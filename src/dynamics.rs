use rand::{Rng, rngs::ThreadRng, thread_rng};
use rand_distr::StandardNormal;

pub trait ExecutionDynamics {
    fn match_prob(&self, offset: f64) -> f64;
}

#[derive(Debug)]
pub struct PoissonRate {
    dt: f64,
    pub scale: f64,
    pub decay: f64,
}

impl PoissonRate {
    pub fn new(dt: f64, scale: f64, decay: f64) -> PoissonRate {
        PoissonRate { dt, scale, decay, }
    }
}

impl ExecutionDynamics for PoissonRate {
    fn match_prob(&self, offset: f64) -> f64 {
        let lambda = self.scale * (-self.decay * offset).exp();

        (lambda * self.dt).max(0.0).min(1.0)
    }
}

impl Default for PoissonRate {
    fn default() -> PoissonRate {
        PoissonRate::new(0.005, 140.0, 1.5)
    }
}

pub trait PriceDynamics {
    fn sample_increment<R: Rng>(&self, rng: &mut R, x: f64) -> f64;
}

#[derive(Debug)]
pub struct BrownianMotion {
    dt: f64,
    pub volatility: f64,
}

impl BrownianMotion {
    pub fn new(dt: f64, volatility: f64) -> BrownianMotion {
        BrownianMotion { dt, volatility, }
    }
}

impl PriceDynamics for BrownianMotion {
    fn sample_increment<R: Rng>(&self, rng: &mut R, _: f64) -> f64 {
        let w: f64 = rng.sample(StandardNormal);

        self.volatility * self.dt.sqrt() * w
    }
}

impl Default for BrownianMotion {
    fn default() -> BrownianMotion {
        BrownianMotion::new(0.005, 2.0)
    }
}

#[derive(Debug)]
pub struct BrownianMotionWithDrift {
    dt: f64,
    pub drift: f64,
    pub volatility: f64,
}

impl BrownianMotionWithDrift {
    pub fn new(dt: f64, drift: f64, volatility: f64) -> BrownianMotionWithDrift {
        BrownianMotionWithDrift { dt, drift, volatility, }
    }
}

impl PriceDynamics for BrownianMotionWithDrift {
    fn sample_increment<R: Rng>(&self, rng: &mut R, _: f64) -> f64 {
        let w: f64 = rng.sample(StandardNormal);

        self.drift * self.dt + self.volatility * self.dt.sqrt() * w
    }
}

impl Default for BrownianMotionWithDrift {
    fn default() -> BrownianMotionWithDrift {
        BrownianMotionWithDrift::new(0.005, 0.0, 2.0)
    }
}

#[derive(Debug)]
pub struct OrnsteinUhlenbeck {
    dt: f64,
    pub rate: f64,
    pub volatility: f64,
}

impl OrnsteinUhlenbeck {
    pub fn new(dt: f64, rate: f64, volatility: f64) -> OrnsteinUhlenbeck {
        OrnsteinUhlenbeck { dt, rate, volatility, }
    }
}

impl PriceDynamics for OrnsteinUhlenbeck {
    fn sample_increment<R: Rng>(&self, rng: &mut R, x: f64) -> f64 {
        let w = BrownianMotion::new(self.dt, self.volatility);

        -self.rate * x * self.dt + w.sample_increment(rng, x)
    }
}

impl Default for OrnsteinUhlenbeck {
    fn default() -> OrnsteinUhlenbeck {
        OrnsteinUhlenbeck::new(1.0, 1.0, 1.0)
    }
}

#[derive(Debug)]
pub struct OrnsteinUhlenbeckWithDrift {
    dt: f64,
    pub rate: f64,
    pub drift: f64,
    pub volatility: f64,
}

impl OrnsteinUhlenbeckWithDrift {
    pub fn new(dt: f64, rate: f64, drift: f64, volatility: f64) -> OrnsteinUhlenbeckWithDrift {
        OrnsteinUhlenbeckWithDrift { dt, rate, drift, volatility, }
    }
}

impl PriceDynamics for OrnsteinUhlenbeckWithDrift {
    fn sample_increment<R: Rng>(&self, rng: &mut R, x: f64) -> f64 {
        let w = BrownianMotion::new(self.dt, self.volatility);

        self.rate * (self.drift - x) * self.dt + w.sample_increment(rng, x)
    }
}

impl Default for OrnsteinUhlenbeckWithDrift {
    fn default() -> OrnsteinUhlenbeckWithDrift {
        OrnsteinUhlenbeckWithDrift::new(1.0, 1.0, 0.0, 1.0)
    }
}

#[derive(Debug)]
pub struct ASDynamics<P, E> {
    rng: ThreadRng,

    pub dt: f64,
    pub time: f64,
    pub price: f64,
    pub price_initial: f64,

    pub price_dynamics: P,
    pub execution_dynamics: E,
}

impl<P, E> ASDynamics<P, E> {
    pub fn new(dt: f64, price: f64, rng: ThreadRng,
               price_dynamics: P, execution_dynamics: E) -> Self
    {
        ASDynamics {
            rng,

            dt,
            time: 0.0,
            price,
            price_initial: price,

            price_dynamics,
            execution_dynamics,
        }
    }
}

impl ASDynamics<BrownianMotionWithDrift, PoissonRate> {
    pub fn default_with_drift(drift: f64) -> Self {
        const DT: f64 = 0.005;

        let pd = BrownianMotionWithDrift::new(DT, drift, 2.0);
        let ed = PoissonRate::new(DT, 140.0, 1.5);

        ASDynamics::new(DT, 100.0, thread_rng(), pd, ed)
    }
}

impl Default for ASDynamics<BrownianMotion, PoissonRate> {
    fn default() -> Self {
        const DT: f64 = 0.005;

        let pd = BrownianMotion::new(DT, 2.0);
        let ed = PoissonRate::new(DT, 140.0, 1.5);

        ASDynamics::new(DT, 100.0, thread_rng(), pd, ed)
    }
}

impl<P, E> ASDynamics<P, E>
where
    P: PriceDynamics,
    E: ExecutionDynamics,
{
    pub fn innovate(&mut self) -> f64 {
        let mut rng = thread_rng();

        let price_inc = self.price_dynamics.sample_increment(&mut rng, self.price);

        self.time += self.dt;
        self.price += price_inc;

        price_inc
    }

    fn try_execute(&mut self, offset: f64) -> Option<f64> {
        let match_prob = self.execution_dynamics.match_prob(offset);

        if self.rng.gen_bool(match_prob) {
            Some(offset)
        } else {
            None
        }
    }

    pub fn try_execute_ask(&mut self, order_price: f64) -> Option<f64> {
        let offset = order_price - self.price;

        self.try_execute(offset)
    }

    pub fn try_execute_bid(&mut self, order_price: f64) -> Option<f64> {
        let offset = self.price - order_price;

        self.try_execute(offset)
    }
}
