#[cfg(test)]
mod tests {

    use rtrec::{identifiers::SerializableValue, slim::SlimMSE};

    #[test]
    fn test_serializable_value_conversion() {
        // Test Integer conversion
        let int_value: Option<SerializableValue> = SerializableValue::from_any(&42);
        assert_eq!(SerializableValue::as_any(&int_value.unwrap()).downcast_ref::<i32>(), Some(&42));

        // Test String conversion
        let string_value = SerializableValue::from_any(&"Hello".to_string());
        assert_eq!(SerializableValue::as_any(&string_value.unwrap()).downcast_ref::<String>(), Some(&"Hello".to_string()));

        let floay_value = SerializableValue::from_any(&3.14); // Float is not supported
        assert!(floay_value.is_none());
    }

    #[test]
    fn test_fit() {
        let mut slim = SlimMSE::new(0.5, 1.0, 0.0002, 0.0001, -5.0, 10.0, None);

        let user_interactions = vec![
            (SerializableValue::Integer(1), SerializableValue::Integer(2), 1620000000.0, 4.0),
            (SerializableValue::Integer(1), SerializableValue::Integer(3), 1620003600.0, 5.0),
        ];

        slim.fit(user_interactions);
    }

    #[test]
    fn test_recommend() {
        // Initialize SlimMSE with sample hyperparameters
        let mut model = SlimMSE::new(0.1, 1.0, 0.001, 0.002, -5.0, 10.0, None);

        // Add some user-item interactions for testing
        let current_time = 0.0; // Starting timestamp for simplicity
        let user_interactions = vec![
            (SerializableValue::Integer(1), SerializableValue::Integer(101), current_time, 5.0), // User 1, Item 101, Rating 5.0
            (SerializableValue::Integer(1), SerializableValue::Integer(102), current_time, 3.0), // User 1, Item 102, Rating 3.0
            (SerializableValue::Integer(2), SerializableValue::Integer(101), current_time, 4.0), // User 2, Item 101, Rating 4.0
            (SerializableValue::Integer(2), SerializableValue::Integer(103), current_time, 2.0), // User 2, Item 103, Rating 2.0
        ];

        // Fit the model with the interactions
        model.fit(user_interactions.clone());

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

}
