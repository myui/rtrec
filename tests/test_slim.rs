use rtrec::slim::SlimMSE;

#[test]
fn test_recommend() {
    // Initialize SlimMSE with sample hyperparameters
    let mut model = SlimMSE::new(0.1, 1.0, 0.001, 0.002, -5.0, 10.0, None);

    // Add some user-item interactions for testing
    let current_time = 0.0; // Starting timestamp for simplicity
    let user_interactions = vec![
        (1, 101, current_time, 5.0),  // User 1, Item 101, Rating 5.0, Timestamp
        (1, 102, current_time, 3.0),  // User 1, Item 102, Rating 3.0, Timestamp
        (2, 101, current_time, 4.0),  // User 2, Item 101, Rating 4.0, Timestamp
        (2, 103, current_time, 2.0),  // User 2, Item 103, Rating 2.0, Timestamp
    ];

    // Fit the model with the interactions including timestamps
    model.fit(user_interactions);

    // Test recommendations for User 1 with top_k = 2
    let recommendations = model.recommend(1, 2, Some(true));

    // Assert that recommendations are not empty
    assert!(!recommendations.is_empty(), "Recommendations should not be empty");

    // Check that recommendations contain valid item IDs
    // Since User 1 has interacted with items 101 and 102, they should not be in the recommendations
    assert!(!recommendations.contains(&101), "Item 101 should not be recommended since user already interacted with it");
    assert!(!recommendations.contains(&102), "Item 102 should not be recommended since user already interacted with it");

    // User 1 has not interacted with item 103, so it should be recommended
    assert!(recommendations.contains(&103), "Item 103 should be recommended");
}