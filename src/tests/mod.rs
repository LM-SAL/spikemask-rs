use crate::detect::Detector;

#[test]
fn a_nearby_spike_appears_on_the_second_pass() {
    let mut image = vec![100.; 32 * 32];
    image[15 * 32 + 16] = 400.;
    image[16 * 32 + 16] = 10500.;
    let detector = Detector {
        width: 32,
        height: 32,
        window: (5, 3),
        sigma_upper: 2.65,
    };
    let once = detector.mask(image.clone(), 1);
    assert!(once[16 * 32 + 16]);
    assert!(!once[15 * 32 + 16]);
    let twice = detector.mask(image, 2);
    assert!(twice[16 * 32 + 16] && twice[15 * 32 + 16]);
}

#[test]
fn an_even_window_split_by_the_cut_is_decided_by_its_median() {
    // Of the samples 0 and 4, only 0 passes the cut for the 4; their median is 2 and sigma is 2.
    let flagged = |sigma_upper| {
        let detector = Detector {
            width: 2,
            height: 1,
            window: (1, 3),
            sigma_upper,
        };
        detector.mask(vec![0., 4.], 1)
    };
    assert_eq!(flagged(0.9), [false, true]);
    assert_eq!(flagged(1.1), [false, false]);
}

#[test]
fn rows_use_the_declared_window_and_skip_missing_neighbors() {
    let detector = Detector {
        width: 11,
        height: 9,
        window: (1, 7),
        sigma_upper: 2.,
    };
    let mut image = vec![0.; 99];
    image[11] = f64::NAN;
    image[14] = 100.;
    assert_eq!(
        detector
            .mask(image, 10)
            .into_iter()
            .filter(|hit| *hit)
            .count(),
        1
    );
}
