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
        let threshold = self.sigma_upper * variance.sqrt();
        // The median passes the cut when over half the samples do; only an even split needs it.
        let passing = 2 * samples.iter().filter(|&&v| value - v > threshold).count();
        if passing != count {
            return passing > count;
        }
        let (left, middle, _) = samples.select_nth_unstable_by(count / 2, f64::total_cmp);
        let median = (left.iter().copied().fold(f64::NEG_INFINITY, f64::max) + *middle) / 2.;
        value - median > threshold
    }

    // Visits each window offset once for LANES adjacent centers, which LLVM evaluates together.
    #[inline(always)]
    fn lanes(&self, image: &[f64], i: usize, mut visit: impl FnMut(&[f64])) {
        let corner = i - self.window.0 / 2 * self.width - self.window.1 / 2;
        for row in 0..self.window.0 {
            for dx in 0..self.window.1 {
                let start = corner + row * self.width + dx;
                visit(&image[start..start + LANES]);
            }
        }
    }

    #[inline(always)]
    fn interior(&self, image: &[f64], i: usize, hits: &mut Vec<usize>, scratch: &mut Vec<f64>) {
        let count = self.window.0 * self.window.1;
        let mut sum = [0.; LANES];
        self.lanes(image, i, |s| {
            for j in 0..LANES {
                sum[j] += s[j];
            }
        });
        let mean = sum.map(|s| s / count as f64);
        let mut variance = [0.; LANES];
        self.lanes(image, i, |s| {
            for j in 0..LANES {
                let delta = s[j] - mean[j];
                variance[j] += delta * delta;
            }
        });
        let threshold = variance.map(|v| self.sigma_upper * (v / count as f64).sqrt());
        let center = &image[i..i + LANES];
        let mut passing = [0; LANES];
        self.lanes(image, i, |s| {
            for j in 0..LANES {
                passing[j] += (center[j] - s[j] > threshold[j]) as usize;
            }
        });
        for j in 0..LANES {
            // A complete window has an odd count, so there is no even split to settle.
            let hit = if sum[j].is_finite() {
                2 * passing[j] > count
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
        let (above, beside) = (self.window.0 / 2, self.window.1 / 2);
        let complete_rows = y >= above && y + above < self.height;
        let mut x = 0;
        while x < self.width {
            if complete_rows && x >= beside && x + LANES + beside <= self.width {
                self.interior(image, start + x, hits, scratch);
                x += LANES;
            } else {
                if self.is_hit(image, start + x, scratch) {
                    hits.push(start + x);
                }
                x += 1;
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
        // Visiting that many neighbors costs more than testing every pixel once.
        if hits.len().saturating_mul(footprint) >= image.len() {
            return self.scan(image);
        }
        // Only windows that overlap a newly masked sample can change on the next pass.
        let mut next: Vec<usize> = hits
            .par_iter()
            .with_min_len(16)
            .map_init(Vec::new, |scratch, &i| {
                let columns = self.columns(i % self.width);
                self.rows(i / self.width)
                    .flat_map(|row| {
                        row * self.width + columns.start()..=row * self.width + columns.end()
                    })
                    .filter(|&j| self.is_hit(image, j, scratch))
                    .collect::<Vec<_>>()
            })
            .flatten_iter()
            .collect();
        // Two masked samples can share a neighbor.
        next.sort_unstable();
        next.dedup();
        next
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
