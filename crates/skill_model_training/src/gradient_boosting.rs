//! Small histogram gradient-boosted regression trees for tabular baselines.
//!
//! Experiment A2 in `docs/experiment-plan-2026-10.md` needs a strong, fast tabular model
//! over per-player match stats. Squared loss only; features are quantile-binned to at most
//! 256 bins once, then every tree is grown depth-wise from per-node gradient histograms,
//! parallel over features. Early stopping watches an optional validation set.

use rayon::prelude::*;

/// Hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct GradientBoostingConfig {
    /// Maximum number of trees.
    pub trees: usize,
    /// Shrinkage applied to every tree.
    pub learning_rate: f32,
    /// Maximum tree depth (root = depth 0).
    pub max_depth: usize,
    /// Minimum rows per leaf.
    pub min_rows_per_leaf: usize,
    /// L2 regularisation on leaf values.
    pub l2_regularisation: f32,
    /// Fraction of rows sampled (without replacement) per tree.
    pub row_subsample: f32,
    /// Stop when validation RMSE has not improved for this many trees.
    pub early_stopping_rounds: usize,
    /// Seed for row subsampling.
    pub seed: u64,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self {
            trees: 2000,
            learning_rate: 0.05,
            max_depth: 6,
            min_rows_per_leaf: 50,
            l2_regularisation: 1.0,
            row_subsample: 0.8,
            early_stopping_rounds: 100,
            seed: 0x5EED,
        }
    }
}

/// A row-major feature matrix.
#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    /// `rows × columns` values, row-major.
    pub values: Vec<f32>,
    /// Number of columns.
    pub columns: usize,
}

impl FeatureMatrix {
    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        match self.values.len().checked_div(self.columns) {
            Some(rows) => rows,
            None => 0,
        }
    }

    fn value(&self, row: usize, column: usize) -> f32 {
        self.values
            .get(row * self.columns + column)
            .copied()
            .unwrap_or(0.0)
    }
}

/// Bin boundaries for one feature: a value goes in bin `i` where `i` = number of
/// boundaries strictly below it.
#[derive(Debug, Clone)]
struct FeatureBins {
    boundaries: Vec<f32>,
}

impl FeatureBins {
    fn fit(values: &mut [f32], maximum_bins: usize) -> Self {
        values.sort_by(f32::total_cmp);
        let mut boundaries = Vec::with_capacity(maximum_bins);
        if values.is_empty() {
            return Self { boundaries };
        }
        for bin in 1..maximum_bins {
            let index = bin * values.len() / maximum_bins;
            if let Some(&candidate) = values.get(index)
                && boundaries.last().is_none_or(|&last| candidate > last)
            {
                boundaries.push(candidate);
            }
        }
        Self { boundaries }
    }

    fn bin(&self, value: f32) -> u8 {
        let index = self
            .boundaries
            .partition_point(|&boundary| boundary < value);
        u8::try_from(index).unwrap_or(u8::MAX)
    }
}

/// Column-major binned copy of a matrix.
struct BinnedColumns {
    columns: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Copy)]
enum Node {
    Leaf {
        value: f32,
    },
    Split {
        feature: usize,
        /// Rows with `bin <= bin_threshold` go left.
        bin_threshold: u8,
        left: usize,
        right: usize,
    },
}

#[derive(Debug, Clone)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn predict_row(&self, binned: &BinnedColumns, row: usize) -> f32 {
        let mut index = 0;
        loop {
            match self.nodes.get(index) {
                Some(Node::Leaf { value }) => return *value,
                Some(Node::Split {
                    feature,
                    bin_threshold,
                    left,
                    right,
                }) => {
                    let bin = binned
                        .columns
                        .get(*feature)
                        .and_then(|column| column.get(row))
                        .copied()
                        .unwrap_or(0);
                    index = if bin <= *bin_threshold { *left } else { *right };
                }
                None => return 0.0,
            }
        }
    }
}

/// A trained ensemble.
#[derive(Debug, Clone)]
pub struct GradientBoostingModel {
    base_prediction: f32,
    learning_rate: f32,
    trees: Vec<Tree>,
    bins: Vec<FeatureBins>,
    /// Total split gain attributed to each feature.
    feature_gain: Vec<f64>,
    /// Validation RMSE after each tree, when a validation set was given.
    validation_curve: Vec<f64>,
}

/// Best split found for one node.
#[derive(Debug, Clone, Copy)]
struct SplitCandidate {
    gain: f64,
    feature: usize,
    bin_threshold: u8,
}

/// Gradient sum and row count of one histogram bin.
#[derive(Debug, Clone, Copy, Default)]
struct HistogramBin {
    gradient: f64,
    count: u32,
}

const fn split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn leaf_value(gradient: f64, count: f64, l2: f64) -> f32 {
    (-gradient / (count + l2)) as f32
}

fn best_split(
    binned: &BinnedColumns,
    rows: &[usize],
    gradients: &[f32],
    config: &GradientBoostingConfig,
) -> Option<SplitCandidate> {
    let l2 = f64::from(config.l2_regularisation);
    let total_gradient: f64 = rows
        .iter()
        .map(|&row| f64::from(gradients.get(row).copied().unwrap_or(0.0)))
        .sum();
    let total_count = rows.len() as f64;
    let parent_score = total_gradient.powi(2) / (total_count + l2);
    let minimum = config.min_rows_per_leaf as u32;

    binned
        .columns
        .par_iter()
        .enumerate()
        .filter_map(|(feature, column)| {
            let mut histogram = [HistogramBin::default(); 256];
            for &row in rows {
                let bin = usize::from(column.get(row).copied().unwrap_or(0));
                if let Some(entry) = histogram.get_mut(bin) {
                    entry.gradient += f64::from(gradients.get(row).copied().unwrap_or(0.0));
                    entry.count += 1;
                }
            }
            let mut left_gradient = 0.0;
            let mut left_count = 0u32;
            let mut best: Option<SplitCandidate> = None;
            for (bin, entry) in histogram.iter().enumerate().take(255) {
                left_gradient += entry.gradient;
                left_count += entry.count;
                let right_count = rows.len() as u32 - left_count;
                if left_count < minimum || right_count < minimum {
                    continue;
                }
                let right_gradient = total_gradient - left_gradient;
                let gain = left_gradient * left_gradient / (f64::from(left_count) + l2)
                    + right_gradient * right_gradient / (f64::from(right_count) + l2)
                    - parent_score;
                if gain > best.map_or(1e-9, |b| b.gain) {
                    best = Some(SplitCandidate {
                        gain,
                        feature,
                        bin_threshold: bin as u8,
                    });
                }
            }
            best
        })
        .max_by(|a, b| a.gain.total_cmp(&b.gain))
}

fn grow_tree(
    binned: &BinnedColumns,
    rows: Vec<usize>,
    gradients: &[f32],
    config: &GradientBoostingConfig,
    feature_gain: &mut [f64],
) -> Tree {
    let l2 = f64::from(config.l2_regularisation);
    let mut nodes = Vec::new();
    // Each pending entry is a node slot to fill, its rows and its depth.
    let mut pending: Vec<PendingNode> = vec![PendingNode {
        index: 0,
        rows,
        depth: 0,
    }];
    nodes.push(Node::Leaf { value: 0.0 });

    while let Some(node) = pending.pop() {
        let gradient: f64 = node
            .rows
            .iter()
            .map(|&row| f64::from(gradients.get(row).copied().unwrap_or(0.0)))
            .sum();
        let leaf = Node::Leaf {
            value: leaf_value(gradient, node.rows.len() as f64, l2),
        };
        let split =
            if node.depth < config.max_depth && node.rows.len() >= 2 * config.min_rows_per_leaf {
                best_split(binned, &node.rows, gradients, config)
            } else {
                None
            };
        let Some(split) = split else {
            if let Some(slot) = nodes.get_mut(node.index) {
                *slot = leaf;
            }
            continue;
        };
        if let Some(gain) = feature_gain.get_mut(split.feature) {
            *gain += split.gain;
        }
        let column = binned.columns.get(split.feature);
        let mut left_rows = Vec::with_capacity(node.rows.len());
        let mut right_rows = Vec::with_capacity(node.rows.len());
        for row in node.rows {
            let goes_left = column
                .and_then(|values| values.get(row))
                .is_some_and(|&bin| bin <= split.bin_threshold);
            if goes_left {
                left_rows.push(row);
            } else {
                right_rows.push(row);
            }
        }
        let left = nodes.len();
        nodes.push(Node::Leaf { value: 0.0 });
        let right = nodes.len();
        nodes.push(Node::Leaf { value: 0.0 });
        if let Some(slot) = nodes.get_mut(node.index) {
            *slot = Node::Split {
                feature: split.feature,
                bin_threshold: split.bin_threshold,
                left,
                right,
            };
        }
        pending.push(PendingNode {
            index: left,
            rows: left_rows,
            depth: node.depth + 1,
        });
        pending.push(PendingNode {
            index: right,
            rows: right_rows,
            depth: node.depth + 1,
        });
    }
    Tree { nodes }
}

/// A node waiting to be split or turned into a leaf.
struct PendingNode {
    index: usize,
    rows: Vec<usize>,
    depth: usize,
}

fn bin_matrix(matrix: &FeatureMatrix, bins: &[FeatureBins]) -> BinnedColumns {
    let columns = (0..matrix.columns)
        .into_par_iter()
        .map(|column| {
            let feature_bins = bins.get(column);
            (0..matrix.rows())
                .map(|row| feature_bins.map_or(0, |b| b.bin(matrix.value(row, column))))
                .collect()
        })
        .collect();
    BinnedColumns { columns }
}

fn rmse(predictions: &[f32], targets: &[f32]) -> f64 {
    let sum: f64 = predictions
        .iter()
        .zip(targets)
        .map(|(p, t)| f64::from(p - t).powi(2))
        .sum();
    (sum / targets.len().max(1) as f64).sqrt()
}

/// A training or validation set.
pub struct Dataset<'a> {
    /// Features.
    pub features: &'a FeatureMatrix,
    /// Regression targets, one per row.
    pub targets: &'a [f32],
}

impl GradientBoostingModel {
    /// Trains on `training`, early-stopping on `validation` when given.
    #[must_use]
    pub fn fit(
        training: &Dataset<'_>,
        validation: Option<&Dataset<'_>>,
        config: &GradientBoostingConfig,
    ) -> Self {
        let columns = training.features.columns;
        let rows = training.features.rows();
        let bins: Vec<FeatureBins> = (0..columns)
            .into_par_iter()
            .map(|column| {
                let mut values: Vec<f32> = (0..rows)
                    .map(|row| training.features.value(row, column))
                    .collect();
                FeatureBins::fit(&mut values, 256)
            })
            .collect();
        let binned = bin_matrix(training.features, &bins);
        let validation_binned = validation.map(|set| bin_matrix(set.features, &bins));

        let base_prediction =
            training.targets.iter().map(|&t| f64::from(t)).sum::<f64>() as f32 / rows.max(1) as f32;
        let mut predictions = vec![base_prediction; rows];
        let mut validation_predictions =
            validation.map(|set| vec![base_prediction; set.features.rows()]);

        let mut trees = Vec::new();
        let mut feature_gain = vec![0.0; columns];
        let mut validation_curve = Vec::new();
        let mut best_rmse = f64::MAX;
        let mut best_tree_count = 0;
        let mut seed = config.seed;

        for _ in 0..config.trees {
            let gradients: Vec<f32> = predictions
                .iter()
                .zip(training.targets)
                .map(|(p, t)| p - t)
                .collect();
            let threshold = (f64::from(config.row_subsample) * u64::MAX as f64) as u64;
            let sampled: Vec<usize> = (0..rows)
                .filter(|_| config.row_subsample >= 1.0 || split_mix(&mut seed) < threshold)
                .collect();
            let tree = grow_tree(&binned, sampled, &gradients, config, &mut feature_gain);

            predictions
                .par_iter_mut()
                .enumerate()
                .for_each(|(row, prediction)| {
                    *prediction = config
                        .learning_rate
                        .mul_add(tree.predict_row(&binned, row), *prediction);
                });
            if let (Some(set), Some(set_binned), Some(set_predictions)) = (
                validation,
                validation_binned.as_ref(),
                validation_predictions.as_mut(),
            ) {
                set_predictions
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(row, prediction)| {
                        *prediction = config
                            .learning_rate
                            .mul_add(tree.predict_row(set_binned, row), *prediction);
                    });
                let current = rmse(set_predictions, set.targets);
                validation_curve.push(current);
                trees.push(tree);
                if current < best_rmse {
                    best_rmse = current;
                    best_tree_count = trees.len();
                } else if trees.len() - best_tree_count >= config.early_stopping_rounds {
                    break;
                }
            } else {
                trees.push(tree);
                best_tree_count = trees.len();
            }
        }
        trees.truncate(best_tree_count);

        Self {
            base_prediction,
            learning_rate: config.learning_rate,
            trees,
            bins,
            feature_gain,
            validation_curve,
        }
    }

    /// Total split gain attributed to each feature: a rough importance ranking.
    #[must_use]
    pub fn feature_gain(&self) -> &[f64] {
        &self.feature_gain
    }

    /// Validation RMSE after each tree, when a validation set was given.
    #[must_use]
    pub fn validation_curve(&self) -> &[f64] {
        &self.validation_curve
    }

    /// Number of trees kept after early stopping.
    #[must_use]
    pub const fn tree_count(&self) -> usize {
        self.trees.len()
    }

    /// Predicts every row of `features`.
    #[must_use]
    pub fn predict(&self, features: &FeatureMatrix) -> Vec<f32> {
        let binned = bin_matrix(features, &self.bins);
        (0..features.rows())
            .into_par_iter()
            .map(|row| {
                self.learning_rate.mul_add(
                    self.trees
                        .iter()
                        .map(|tree| tree.predict_row(&binned, row))
                        .sum::<f32>(),
                    self.base_prediction,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(rows: usize, seed: u64) -> (FeatureMatrix, Vec<f32>) {
        let mut state = seed;
        let mut values = Vec::with_capacity(rows * 3);
        let mut targets = Vec::with_capacity(rows);
        for _ in 0..rows {
            let a = (split_mix(&mut state) % 1000) as f32 / 1000.0;
            let b = (split_mix(&mut state) % 1000) as f32 / 1000.0;
            let noise = (split_mix(&mut state) % 1000) as f32 / 1000.0;
            values.extend_from_slice(&[a, b, noise]);
            // Non-linear in `a`, interaction with `b`, the third column is pure noise.
            targets.push((5.0 * a).mul_add(b, if a > 0.5 { 10.0 } else { 0.0 }));
        }
        (FeatureMatrix { values, columns: 3 }, targets)
    }

    #[test]
    fn learns_a_nonlinear_function() {
        let (train_x, train_y) = synthetic(4000, 1);
        let (valid_x, valid_y) = synthetic(1000, 2);
        let config = GradientBoostingConfig {
            trees: 300,
            min_rows_per_leaf: 10,
            ..GradientBoostingConfig::default()
        };
        let model = GradientBoostingModel::fit(
            &Dataset {
                features: &train_x,
                targets: &train_y,
            },
            Some(&Dataset {
                features: &valid_x,
                targets: &valid_y,
            }),
            &config,
        );
        let predictions = model.predict(&valid_x);
        let error = rmse(&predictions, &valid_y);
        // Target standard deviation is ~5.2; a decent fit is far below that.
        assert!(error < 0.5, "rmse {error}");
        // The noise column must carry the least gain.
        assert!(model.feature_gain[2] < model.feature_gain[0]);
        assert!(model.feature_gain[2] < model.feature_gain[1]);
    }

    #[test]
    fn early_stopping_truncates_to_the_best_tree() {
        let (train_x, train_y) = synthetic(500, 3);
        let (valid_x, valid_y) = synthetic(500, 4);
        let config = GradientBoostingConfig {
            trees: 400,
            learning_rate: 0.5,
            min_rows_per_leaf: 2,
            early_stopping_rounds: 20,
            ..GradientBoostingConfig::default()
        };
        let model = GradientBoostingModel::fit(
            &Dataset {
                features: &train_x,
                targets: &train_y,
            },
            Some(&Dataset {
                features: &valid_x,
                targets: &valid_y,
            }),
            &config,
        );
        let best = model
            .validation_curve
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(index, _)| index + 1);
        assert_eq!(model.tree_count(), best);
    }
}

impl GradientBoostingModel {
    /// Converts to the binning-free form `ml_model` ships for inference.
    ///
    /// A split on bin `t` sends a row left when `bin ≤ t`, and `bin` counts the boundaries
    /// strictly below the value, so `bin ≤ t ⇔ value ≤ boundaries[t]`. A split past the last
    /// boundary sends everything left (`f32::MAX`, which JSON can represent).
    #[must_use]
    pub fn freeze(&self) -> skill_model::FrozenEnsemble {
        use skill_model::FrozenNode;
        let trees = self
            .trees
            .iter()
            .map(|tree| {
                tree.nodes
                    .iter()
                    .map(|node| match *node {
                        Node::Leaf { value } => FrozenNode::Leaf { value },
                        Node::Split {
                            feature,
                            bin_threshold,
                            left,
                            right,
                        } => FrozenNode::Split {
                            feature,
                            threshold: self
                                .bins
                                .get(feature)
                                .and_then(|bins| bins.boundaries.get(usize::from(bin_threshold)))
                                .copied()
                                .unwrap_or(f32::MAX),
                            left,
                            right,
                        },
                    })
                    .collect()
            })
            .collect();
        skill_model::FrozenEnsemble {
            base_prediction: self.base_prediction,
            learning_rate: self.learning_rate,
            trees,
        }
    }
}

#[cfg(test)]
mod freeze_tests {
    use super::*;

    /// The frozen ensemble must reproduce the binned predictions exactly.
    #[test]
    fn frozen_predictions_match_binned_predictions() {
        let mut state = 7u64;
        let rows = 2000;
        let mut values = Vec::with_capacity(rows * 2);
        let mut targets = Vec::with_capacity(rows);
        for _ in 0..rows {
            let a = (split_mix(&mut state) % 10_000) as f32 / 100.0;
            let b = (split_mix(&mut state) % 10_000) as f32 / 100.0;
            values.extend_from_slice(&[a, b]);
            targets.push(if a > 40.0 { b } else { -b });
        }
        let matrix = FeatureMatrix { values, columns: 2 };
        let model = GradientBoostingModel::fit(
            &Dataset {
                features: &matrix,
                targets: &targets,
            },
            None,
            &GradientBoostingConfig {
                trees: 50,
                min_rows_per_leaf: 5,
                ..GradientBoostingConfig::default()
            },
        );
        let frozen = model.freeze();
        let binned = model.predict(&matrix);
        for (row, (features, expected)) in matrix
            .values
            .as_chunks::<2>()
            .0
            .iter()
            .zip(&binned)
            .enumerate()
        {
            let frozen_prediction = frozen.predict(features);
            assert!(
                (frozen_prediction - expected).abs() < 1e-3,
                "row {row}: {frozen_prediction} vs {expected}"
            );
        }
    }
}
