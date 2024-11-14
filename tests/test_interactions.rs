#[cfg(test)]
mod tests {

    use rtrec::interactions::UserItemInteractions;
    use std::thread::sleep;
    use std::time::{Duration, SystemTime};
    use std::f32::consts::E;

    #[test]
    #[should_panic(expected = "max_value should be greater than min_value")]
    fn test_initialization_invalid_bounds() {
        UserItemInteractions::new(10.0, -5.0, None); // Should panic
    }

    #[test]
    fn test_add_interaction() {
        let mut interactions = UserItemInteractions::new(-5.0, 10.0, None);
        let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f32();

        interactions.add_interaction(1, 100, current_time, 3.0, false); // User 1, Item 100, delta 3.0
        assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 3.0);

        interactions.add_interaction(1, 100, current_time, 5.0, false); // Modify interaction
        assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 8.0); // Expect 8.0

        interactions.add_interaction(1, 100, current_time, 5.0, false); // Clamping above max
        assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 10.0); // Should be clamped to max_value

        interactions.add_interaction(1, 100, current_time, -100.0, false); // Clamping below min
        assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), -5.0); // Should be clamped to min_value
    }

    #[test]
    fn test_get_user_items() {
        let mut interactions = UserItemInteractions::new(-5.0, 10.0, None);
        let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f32();

        interactions.add_interaction(1, 100, current_time + 1.0, 3.0, false);
        interactions.add_interaction(1, 101, current_time + 2.0, 2.0, false);

        let items = interactions.get_user_items(1, None);
        assert_eq!(items.len(), 2);
        assert!(items.contains(&100));
        assert!(items.contains(&101));

        let items = interactions.get_user_items(1, Some(10));
        assert_eq!(items.len(), 2);
        assert!(items.contains(&100));
        assert!(items.contains(&101));

        let items = interactions.get_user_items(1, Some(1));
        assert_eq!(items.len(), 1);
        assert!(items.contains(&100));
    }

    #[test]
    fn test_add_interaction_upsert() {
        let mut interactions = UserItemInteractions::new(-5.0, 10.0, None);
        let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f32();

        interactions.add_interaction(1, 100, current_time, 3.0, true); // User 1, Item 100, delta 3.0
        assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 3.0);

        interactions.add_interaction(1, 100, current_time, -3.0, true); // User 1, Item 100, delta 3.0
        assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), -3.0);
    }

    #[test]
    fn test_get_all_non_interacted_items() {
        let mut interactions = UserItemInteractions::new(-5.0, 10.0, None);
        let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f32();

        interactions.add_interaction(1, 100, current_time, 3.0, false);
        interactions.add_interaction(1, 101, current_time, 2.0, false);

        let non_interacted_items = interactions.get_all_non_interacted_items(1);
        assert!(!non_interacted_items.contains(&100));
        assert!(!non_interacted_items.contains(&101));

        // Adding an item that the user has not interacted with
        interactions.add_interaction(2, 200, current_time, 1.0, false); // User 2, Item 200
        interactions.add_interaction(2, 101, current_time, 1.0, false); // User 2, Item 200

        let non_interacted_items_user_2 = interactions.get_all_non_interacted_items(2);
        assert!(non_interacted_items_user_2.contains(&100));
        assert!(!non_interacted_items_user_2.contains(&101));
    }

    #[test]
    fn test_get_all_non_negative_items() {
        let mut interactions = UserItemInteractions::new(-5.0, 10.0, None);
        let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f32();

        interactions.add_interaction(1, 100, current_time, 3.0, false); // User 1, Item 100, delta 3.0
        interactions.add_interaction(1, 101, current_time, -5.0, false); // User 1, Item 101, delta -5.0 (will clamp to min_value)

        let non_negative_items = interactions.get_all_non_negative_items(1);
        assert_eq!(non_negative_items.len(), 1);
        assert!(non_negative_items.contains(&100)); // Only item 100 should be non-negative
        assert!(!non_negative_items.contains(&101)); // Item 101 should not be included as its value is -5.0
    }

    #[test]
    fn test_decay_with_decay_rate() {
        let mut interactions = UserItemInteractions::new(-5.0, 10.0, Some(7_f32)); // Decay rate of 7 days
        let decay_rate = 1.0_f32 - (E.ln() / 7_f32);
        let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f32();

        let tstamp_7_days_ago = current_time - (7_f32 * 86400_f32);
        interactions.add_interaction(1, 100, tstamp_7_days_ago, 5.0, false); // User 1, Item 100, delta 5.0
        let actual_rating = interactions.get_user_item_rating(1, 100, 0.0);

        let elapsed_days = 7_f32;  // 7 days
        let decay_factor = decay_rate.powf(elapsed_days);
        let expected_rating1 = 5.0_f32 * decay_factor;

        assert!((actual_rating - expected_rating1).abs() < 1e-5, "Expected: {}, Actual: {}", expected_rating1, actual_rating);

        // Sleep for 2 seconds to allow decay to occur
        sleep(Duration::from_secs(2));
        let actual_rating = interactions.get_user_item_rating(1, 100, 0.0);

        let elapsed_days = 7_f32 + 2_f32 / 86400.0_f32;  // 7 days + 2 second in days
        let decay_factor = decay_rate.powf(elapsed_days);
        let expected_rating2 = 5.0_f32 * decay_factor;

        assert!((actual_rating - expected_rating2).abs() < 1e-5, "Expected: {}, Actual: {}", expected_rating2, actual_rating);
        assert!(expected_rating1 > expected_rating2, "Expected rating should be greater than the rating after 2 seconds");
    }

    #[test]
    fn test_decay_without_decay_rate() {
        let mut interactions = UserItemInteractions::new(-5.0, 10.0, None); // No decay rate
        let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f32();

        interactions.add_interaction(1, 100, current_time, 5.0, false); // User 1, Item 100, delta 5.0
        assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 5.0);

        // Sleep for 2 seconds, but should not affect the value since decay is not applied
        sleep(Duration::from_secs(2));

        assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 5.0); // Value should still be 5.0
    }

}