//! Iterative upper sigma clipping on a single detector image.
use rayon::prelude::*;

const LANES: usize = 8;

pub(crate) struct Detector {
    pub(crate) height: usize,
    pub(crate) width: usize,
    pub(crate) window: (usize, usize),
    pub(crate) sigma_upper: f64,
}

impl Detector {
    fn rows(&self, y: usize) -> std::ops::RangeInclusive<usize> {
        let radius = self.window.0 / 2;
        y.saturating_sub(radius)..=y.saturating_add(radius).min(self.height - 1)
    }

    fn columns(&self, x: usize) -> std::ops::RangeInclusive<usize> {
        let radius = self.window.1 / 2;
        x.saturating_sub(radius)..=x.saturating_add(radius).min(self.width - 1)
    }

    fn is_hit(&self, image: &[f64], i: usize, samples: &mut Vec<f64>) -> bool {
        let value = image[i];
        if !value.is_finite() {
            return false;
        }
        let columns = self.columns(i % self.width);
        samples.clear();
        for row in self.rows(i / self.width) {
            let offset = row * self.width;
            samples.extend(
                image[offset + columns.start()..=offset + columns.end()]
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite()),
            );
        }
        let count = samples.len();
        let mean = samples.iter().sum::<f64>() / count as f64;
        let variance = samples.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / count as f64;
        let (left, middle, _) = samples.select_nth_unstable_by(count / 2, f64::total_cmp);
        let median = if count.is_multiple_of(2) {
            (left.iter().copied().fold(f64::NEG_INFINITY, f64::max) + *middle) / 2.
        } else {
            *middle
        };
        value - median > self.sigma_upper * variance.sqrt()
    }

    // The fixed shape lets LLVM evaluate adjacent centers together.
    #[inline(always)]
    fn interior(&self, image: &[f64], i: usize, hits: &mut Vec<usize>, scratch: &mut Vec<f64>) {
        let mut samples = [[0.; LANES]; 15];
        let mut sum = [0.; LANES];
        for row in 0..5 {
            let start = i - 2 * self.width - 1 + row * self.width;
            for dx in 0..3 {
                let s = &mut samples[3 * row + dx];
                s.copy_from_slice(&image[start + dx..start + dx + LANES]);
                for j in 0..LANES {
                    sum[j] += s[j];
                }
            }
        }
        let mean = sum.map(|s| s / 15.);
        let mut variance = [0.; LANES];
        for sample in &samples {
            for j in 0..LANES {
                let delta = sample[j] - mean[j];
                variance[j] += delta * delta;
            }
        }
        let threshold = variance.map(|v| self.sigma_upper * (v / 15.).sqrt());
        let mut above = [0u64; LANES];
        // The eighth ordered sample is the median: eight passing comparisons give the same strict cut.
        for sample in &samples {
            for j in 0..LANES {
                above[j] += (samples[7][j] - sample[j] > threshold[j]) as u64;
            }
        }
        for j in 0..LANES {
            let hit = if sum[j].is_finite() {
                above[j] >= 8
            } else {
                self.is_hit(image, i + j, scratch)
            };
            if hit {
                hits.push(i + j);
            }
        }
    }

    #[inline(always)]
    fn row(&self, image: &[f64], y: usize, hits: &mut Vec<usize>, scratch: &mut Vec<f64>) {
        let start = y * self.width;
        let mut x = 0;
        if self.window == (5, 3) && y >= 2 && y + 2 < self.height && self.width >= LANES + 2 {
            if self.is_hit(image, start, scratch) {
                hits.push(start);
            }
            x = 1;
            while x + LANES < self.width {
                self.interior(image, start + x, hits, scratch);
                x += LANES;
            }
        }
        for col in x..self.width {
            if self.is_hit(image, start + col, scratch) {
                hits.push(start + col);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn row_avx2(
        &self,
        image: &[f64],
        y: usize,
        hits: &mut Vec<usize>,
        scratch: &mut Vec<f64>,
    ) {
        self.row(image, y, hits, scratch);
    }

    fn scan(&self, image: &[f64]) -> Vec<usize> {
        (0..self.height)
            .into_par_iter()
            .with_min_len(4)
            .fold(
                || (Vec::new(), Vec::new()),
                |(mut hits, mut scratch), y| {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") {
                        // SAFETY: runtime detection guards the only CPU-specific entry point.
                        unsafe { self.row_avx2(image, y, &mut hits, &mut scratch) };
                        return (hits, scratch);
                    }
                    self.row(image, y, &mut hits, &mut scratch);
                    (hits, scratch)
                },
            )
            .map(|(hits, _)| hits)
            .reduce(Vec::new, |mut left, right| {
                left.extend(right);
                left
            })
    }

    fn rescan(&self, image: &[f64], hits: &[usize]) -> Vec<usize> {
        let footprint = self
            .window
            .0
            .min(self.height)
            .saturating_mul(self.window.1.min(self.width));
        let visits = hits.len().saturating_mul(footprint);
        if visits >= image.len() {
            return self.scan(image);
        }
        // Only windows that overlap a newly masked sample can change on the next pass.
        let mut candidates = Vec::with_capacity(visits);
        for &i in hits {
            let cols = self.columns(i % self.width);
            for row in self.rows(i / self.width) {
                let base = row * self.width;
                candidates.extend(
                    cols.clone()
                        .map(|x| base + x)
                        .filter(|&j| image[j].is_finite()),
                );
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
            .into_par_iter()
            .with_min_len(128)
            .map_init(Vec::new, |scratch, i| {
                self.is_hit(image, i, scratch).then_some(i)
            })
            .flatten()
            .collect()
    }

    pub(crate) fn mask(&self, mut image: Vec<f64>, max_iters: usize) -> Vec<bool> {
        let mut mask = vec![false; image.len()];
        if image.is_empty() || self.window == (1, 1) {
            return mask;
        }
        let mut hits = self.scan(&image);
        for pass in 0..max_iters {
            if hits.is_empty() {
                break;
            }
            for &i in &hits {
                mask[i] = true;
                image[i] = f64::NAN;
            }
            if pass + 1 < max_iters {
                hits = self.rescan(&image, &hits);
            }
        }
        mask
    }
}
