#[cfg(test)]
mod tests {

    use rtrec::{identifiers::SerializableValue, slim::SlimMSE};
    use rand::prelude::SliceRandom;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use approx::assert_relative_eq;

    #[test]
    fn test_serializable_value_conversion() {
        // Test Integer conversion
        let int_value: SerializableValue = SerializableValue::from_any(&42);
        assert_eq!(SerializableValue::as_any(&int_value).downcast_ref::<i32>(), Some(&42));

        // Test String conversion
        let string_value = SerializableValue::from_any(&"Hello".to_string());
        assert_eq!(SerializableValue::as_any(&string_value).downcast_ref::<String>(), Some(&"Hello".to_string()));
    }

    #[test]
    #[should_panic(expected = "Unsupported type for SerializableValue")]
    fn test_from_any_with_unsupported_type() {
        let value: f64 = 2.13;

        // This should panic because f64 is not supported
        SerializableValue::from_any(&value);
    }

    #[test]
    fn test_fit() {
        let mut slim = SlimMSE::new("adagrad", 0.5, 0.0002, 0.0001, (-5.0, 10.0), None, None);

        let user_interactions = vec![
            (SerializableValue::Integer(1), SerializableValue::Integer(2), 1620000000.0, 4.0),
            (SerializableValue::Integer(1), SerializableValue::Integer(3), 1620003600.0, 5.0),
        ];

        slim.fit(user_interactions, Some(false));
    }

    #[test]
    fn test_fit_converges() {
        let mut slim = SlimMSE::new("adagrad", 0.5, 0.0002, 0.0001, (-5.0, 10.0), None, None);

        let user_interactions = vec![
            (SerializableValue::Integer(1), SerializableValue::Integer(2), 1620000000.0, 4.0),
            (SerializableValue::Integer(1), SerializableValue::Integer(3), 1620003600.0, 5.0),
            (SerializableValue::Integer(2), SerializableValue::Integer(3), 1620003600.0, 4.0),
        ];

        for i in 0..10 {
            slim.fit(user_interactions.clone(), Some(i > 0));
        }
        let err1 = slim.get_empirical_error();
        for _ in 0..10 {
            slim.fit(user_interactions.clone(), Some(true));
        }
        let err2 = slim.get_empirical_error();
        assert!(err2 < err1, "Empirical error should decrease after fitting more data");
    }

    #[test]
    fn test_recommend() {
        // Initialize SlimMSE with sample hyperparameters
        let mut model = SlimMSE::new("adagrad", 0.1, 0.0002, 0.0001, (-5.0, 10.0), None, None);

        // Add some user-item interactions for testing
        let current_time = 0.0; // Starting timestamp for simplicity
        let user_interactions = vec![
            (SerializableValue::Integer(1), SerializableValue::Integer(101), current_time, 5.0), // User 1, Item 101, Rating 5.0
            (SerializableValue::Integer(1), SerializableValue::Integer(102), current_time, 3.0), // User 1, Item 102, Rating 3.0
            (SerializableValue::Integer(2), SerializableValue::Integer(101), current_time, 4.0), // User 2, Item 101, Rating 4.0
            (SerializableValue::Integer(2), SerializableValue::Integer(103), current_time, 2.0), // User 2, Item 103, Rating 2.0
        ];

        // Fit the model with the interactions
        model.fit(user_interactions.clone(), Some(false));

        // Test recommendations for User 1 with top_k = 2
        let recommendations = model.recommend(SerializableValue::Integer(1), 2, Some(true));

        // Assert that recommendations are not empty
        assert!(!recommendations.is_empty(), "Recommendations should not be empty");

        // Check that recommendations contain valid item IDs
        // Since User 1 has interacted with items 101 and 102, they should not be in the recommendations
        assert!(!recommendations.contains(&SerializableValue::Integer(101)), "Item 101 should not be recommended since user already interacted with it");
        assert!(!recommendations.contains(&SerializableValue::Integer(102)), "Item 102 should not be recommended since user already interacted with it");

        // User 1 has not interacted with item 103, so it should be recommended
        assert!(recommendations.contains(&SerializableValue::Integer(103)), "Item 103 should be recommended");
    }

    #[test]
    fn test_similar_items() {
        // Initialize SlimMSE with sample hyperparameters
        let mut model = SlimMSE::new("adagrad", 0.5, 0.0002, 0.0001, (-5.0, 10.0), None, None);

        // Define current time for simplicity (using a fixed timestamp)
        let current_time = 0.0;

        // Define user interactions similar to the Python test
        let mut user_interactions: Vec<(SerializableValue, SerializableValue, f32, f32)> = vec![
            (SerializableValue::Text("a".to_string()), SerializableValue::Integer(1), current_time, 5.0),
            (SerializableValue::Text("a".to_string()), SerializableValue::Integer(2), current_time, 2.0),
            (SerializableValue::Text("a".to_string()), SerializableValue::Integer(3), current_time, 3.0),
            (SerializableValue::Text("b".to_string()), SerializableValue::Integer(1), current_time, 4.0),
            (SerializableValue::Text("b".to_string()), SerializableValue::Integer(3), current_time, 2.0),
            (SerializableValue::Text("c".to_string()), SerializableValue::Integer(2), current_time, 3.0),
            (SerializableValue::Text("c".to_string()), SerializableValue::Integer(3), current_time, 4.0),
            (SerializableValue::Text("d".to_string()), SerializableValue::Integer(4), current_time, 4.0),
            (SerializableValue::Text("d".to_string()), SerializableValue::Integer(5), current_time, 5.0),
        ];

        // Shuffle and fit interactions multiple times to simulate randomized training as in the Python test
        let mut rng: StdRng = StdRng::seed_from_u64(43);
        for i in 0..10 {
            user_interactions.shuffle(&mut rng);
            model.fit(user_interactions.clone(), Some(i > 0));
        }

        assert_relative_eq!(model.predict_rating(SerializableValue::Text("d".to_string()), SerializableValue::Integer(4)), 4.0, epsilon=0.01);

        // Define query items and parameters for the similar items search
        let query_items = vec![SerializableValue::Integer(1), SerializableValue::Integer(2)];
        let top_k = 2;

        // Get similar items
        let similar_items = model.similar_items(query_items.clone(), top_k, true);

        // Ensure similar_items has the expected length for each query item
        assert_eq!(similar_items.len(), query_items.len(), "Each query item should have a list of similar items");

        // Check that each item has the expected number of similar items (<= top_k)
        for similar in &similar_items {
            assert!(similar.len() <= top_k, "Each query item should return no more than top_k similar items");
        }

        // Assert similar items for specific query items, matching expected Python results
        // Note: Adjust item IDs if expected results differ in Rust implementation
        assert_eq!(similar_items[0], vec![SerializableValue::Integer(3), SerializableValue::Integer(2)], "Expected similar items for item 1 are 3 and 2");
        assert_eq!(similar_items[1], vec![SerializableValue::Integer(3), SerializableValue::Integer(1)], "Expected similar items for item 2 are 3 and 1");
    }

}
