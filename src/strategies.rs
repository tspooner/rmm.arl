#[derive(Debug)]
pub struct LinearUtilityStrategy {
    k: f64,
}

impl LinearUtilityStrategy {
    pub fn new(k: f64) -> LinearUtilityStrategy {
        LinearUtilityStrategy { k, }
    }

    pub fn compute(&self, _: f64, _: f64, _: f64) -> [f64; 2] {
        [1.0 / self.k, 1.0 / self.k]
    }
}

#[derive(Debug)]
pub struct LinearUtilityTerminalPenaltyStrategy {
    k: f64,
    eta: f64,
}

impl LinearUtilityTerminalPenaltyStrategy {
    pub fn new(k: f64, eta: f64) -> LinearUtilityTerminalPenaltyStrategy {
        LinearUtilityTerminalPenaltyStrategy { k, eta, }
    }

    pub fn compute(&self, _: f64, price: f64, inventory: f64) -> [f64; 2] {
        let rp = price - 2.0 * inventory * self.eta;
        let sp = 2.0 / self.k + self.eta;

        [rp + sp / 2.0 - price, price - (rp - sp / 2.0)]
    }
}

#[derive(Debug)]
pub struct ExponentialUtilityStrategy {
    k: f64,
    gamma: f64,
    volatility: f64,
}

impl ExponentialUtilityStrategy {
    pub fn new(k: f64, gamma: f64, volatility: f64) -> ExponentialUtilityStrategy {
        ExponentialUtilityStrategy { k, gamma, volatility, }
    }

    pub fn compute(&self, time: f64, price: f64, inventory: f64) -> [f64; 2] {
        let gss = self.gamma * self.volatility * self.volatility;

        let rp = price - inventory * gss * (1.0 - time);
        let sp = gss * (1.0 - time) + (2.0 / self.gamma) * (1.0 + self.gamma / self.k).ln();

        [rp + sp / 2.0 - price, price - (rp - sp / 2.0)]
    }
}
