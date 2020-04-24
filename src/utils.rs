pub fn mean_var(values: &[f64]) -> [f64; 2] {
    let n = values.len() as f64;

    let sum: f64 = values.iter().sum();
    let sumsq: f64 = values.iter().fold(0.0, |acc, v| acc + v * v);

    let mean = sum / n;
    let var = sumsq / n - mean * mean;

    [mean, var]
}

pub fn median_quantiles(values: &[f64]) -> [f64; 3] {
    let pivot = values.len() / 4;

    [values[pivot], values[pivot * 2], values[pivot * 3]]
}

#[derive(Clone, Copy, Debug)]
pub struct Estimate(pub f64, pub f64);

impl Estimate {
    pub fn from_slice(values: &[f64]) -> Self {
        let [mean, var] = mean_var(values);

        Estimate(mean, var.sqrt())
    }
}

impl slog::Value for Estimate {
    fn serialize(
        &self,
        _rec: &slog::Record,
        key: slog::Key,
        serializer: &mut dyn slog::Serializer
    ) -> slog::Result
    {
        serializer.emit_arguments(key, &format_args!("{} ± {}", self.0, self.1))
    }
}
