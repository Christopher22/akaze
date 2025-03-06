use akaze::Akaze;
use arrsac::Arrsac;
use bitarray::{BitArray, Hamming};
use cv_core::nalgebra::{Point2, Vector2};
use cv_core::sample_consensus::Consensus;
use cv_core::{CameraModel, FeatureMatch};
use log::*;
use old_rand::rngs::StdRng;
use old_rand::SeedableRng;
use space::{Knn, KnnFromMetricAndBatch, LinearKnn};
use std::path::Path;

const LOWES_RATIO: f32 = 0.5;

type Descriptor = BitArray<64>;
type Match = FeatureMatch<cv_pinhole::NormalizedKeyPoint>;

fn image_to_kps(path: impl AsRef<Path>) -> (Vec<akaze::KeyPoint>, Vec<Descriptor>) {
    Akaze::sparse().extract_path(path).unwrap()
}

#[test]
fn estimate_pose() {
    pretty_env_logger::init_timed();
    // Intrinsics retrieved from calib_cam_to_cam.txt K_00.
    let intrinsics = cv_pinhole::CameraIntrinsics {
        focals: Vector2::new(9.842_439e2, 9.808_141e2),
        principal_point: Point2::new(6.9e2, 2.331_966e2),
        skew: 0.0,
    };

    // Extract features with AKAZE.
    info!("Extracting features");
    let (kps1, ds1) = image_to_kps("res/0000000000.png");
    let (kps2, ds2) = image_to_kps("res/0000000014.png");

    // This ensures the underlying algorithm does not change
    // by making sure that we get the exact expected number of features.
    assert_eq!(ds1.len(), 575);
    assert_eq!(ds2.len(), 497);

    // Perform matching.
    info!(
        "Running matching on {} and {} descriptors",
        ds1.len(),
        ds2.len()
    );
    let matches: Vec<Match> = match_descriptors(&ds1, &ds2)
        .into_iter()
        .map(|(ix1, ix2)| {
            let a = intrinsics.calibrate(kps1[ix1]);
            let b = intrinsics.calibrate(kps2[ix2]);
            FeatureMatch(a, b)
        })
        .collect();
    info!("Finished matching with {} matches", matches.len());
    assert_eq!(matches.len(), 30);

    // Run ARRSAC with the eight-point algorithm.
    info!("Running ARRSAC");

    let mut arrsac = Arrsac::new(0.001, StdRng::from_seed([1; 32]));
    let eight_point = eight_point::EightPoint::new();
    let (_, inliers) = arrsac
        .model_inliers(&eight_point, matches.iter().copied())
        .expect("failed to estimate model");
    info!("inliers: {}", inliers.len());
    info!(
        "inlier ratio: {}",
        inliers.len() as f32 / matches.len() as f32
    );

    // Ensures the underlying algorithms don't change at all.
    assert_eq!(inliers.len(), 30);
}

fn match_descriptors(a: &[BitArray<64>], b: &[BitArray<64>]) -> Vec<(usize, usize)> {
    // The best match for each feature in frame a to frame b's features.
    let forward_matches = matching(a, b);
    // The best match for each feature in frame b to frame a's features.
    let reverse_matches = matching(b, a);
    forward_matches
        .into_iter()
        .enumerate()
        .filter_map(move |(aix, bix)| {
            // First we only proceed if there was a sufficient bix match.
            // Filter out matches which are not symmetric.
            // Symmetric is defined as the best and sufficient match of a being b,
            // and likewise the best and sufficient match of b being a.
            bix.map(|bix| [aix, bix])
                .filter(|&[aix, bix]| reverse_matches[bix] == Some(aix))
        })
        .map(|[aix, bix]| (aix, bix))
        .collect()
}

fn matching(a_descriptors: &[Descriptor], b_descriptors: &[Descriptor]) -> Vec<Option<usize>> {
    let points: Vec<_> = b_descriptors
        .iter()
        .enumerate()
        .map(|(i, x)| (x.clone(), i))
        .collect();
    let knn_b = LinearKnn::from_metric_and_batch(Hamming, points.iter());

    (0..a_descriptors.len())
        .map(|a_feature| {
            let knn = knn_b.knn(&a_descriptors[a_feature], 2);
            if (knn[0].0.distance + 24 < knn[1].0.distance)
                && (knn[0].0.distance as f32) < knn[1].0.distance as f32 * LOWES_RATIO
            {
                Some(knn[0].0.index)
            } else {
                None
            }
        })
        .collect()
}
