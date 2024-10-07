use rtrec::interactions::UserItemInteractions;
use std::thread::sleep;
use std::time::{Duration, SystemTime};

#[test]
#[should_panic(expected = "max_value should be greater than min_value")]
fn test_initialization_invalid_bounds() {
    UserItemInteractions::new(10.0, -5.0, None); // Should panic
}

#[test]
fn test_add_interaction() {
    let mut interactions = UserItemInteractions::new(-5.0, 10.0, None);
    let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f32;

    interactions.add_interaction(1, 100, current_time, 3.0); // User 1, Item 100, delta 3.0
    assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 3.0);

    interactions.add_interaction(1, 100, current_time, 5.0); // Modify interaction
    assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 8.0); // Expect 8.0

    interactions.add_interaction(1, 100, current_time, 5.0); // Clamping above max
    assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 10.0); // Should be clamped to max_value

    interactions.add_interaction(1, 100, current_time, -100.0); // Clamping below min
    assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), -5.0); // Should be clamped to min_value
}

#[test]
fn test_get_all_items_for_user() {
    let mut interactions = UserItemInteractions::new(-5.0, 10.0, None);
    let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f32;

    interactions.add_interaction(1, 100, current_time, 3.0);
    interactions.add_interaction(1, 101, current_time, 2.0);

    let items = interactions.get_all_items_for_user(1);
    assert_eq!(items.len(), 2);
    assert!(items.contains(&100));
    assert!(items.contains(&101));
}

#[test]
fn test_get_all_non_interacted_items() {
    let mut interactions = UserItemInteractions::new(-5.0, 10.0, None);
    let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f32;

    interactions.add_interaction(1, 100, current_time, 3.0);
    interactions.add_interaction(1, 101, current_time, 2.0);

    let non_interacted_items = interactions.get_all_non_interacted_items(1);
    assert!(!non_interacted_items.contains(&100));
    assert!(!non_interacted_items.contains(&101));

    // Adding an item that the user has not interacted with
    interactions.add_interaction(2, 200, current_time, 1.0); // User 2, Item 200
    let non_interacted_items_user_2 = interactions.get_all_non_interacted_items(2);
    assert!(non_interacted_items_user_2.contains(&200)); // Item 200 should be in non-interacted for User 2
}

#[test]
fn test_get_all_non_negative_items() {
    let mut interactions = UserItemInteractions::new(-5.0, 10.0, None);
    let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f32;

    interactions.add_interaction(1, 100, current_time, 3.0); // User 1, Item 100, delta 3.0
    interactions.add_interaction(1, 101, current_time, -5.0); // User 1, Item 101, delta -5.0 (will clamp to min_value)

    let non_negative_items = interactions.get_all_non_negative_items(1);
    assert_eq!(non_negative_items.len(), 1);
    assert!(non_negative_items.contains(&100)); // Only item 100 should be non-negative
    assert!(!non_negative_items.contains(&101)); // Item 101 should not be included as its value is -5.0
}

#[test]
fn test_decay_with_decay_rate() {
    let mut interactions = UserItemInteractions::new(-5.0, 10.0, Some(0.1)); // 10% decay rate
    let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f32;

    interactions.add_interaction(1, 100, current_time, 5.0); // User 1, Item 100, delta 5.0
    assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 5.0);

    // Sleep for 2 seconds to allow decay to occur
    sleep(Duration::from_secs(2));

    // Apply decay: expected new value should be 5.0 * (1 - 0.1) ^ 2
    let expected_value = 5.0_f32 * (1.0_f32 - 0.1_f32).powi(2);
    assert!((interactions.get_user_item_rating(1, 100, 0.0) - expected_value).abs() < 1e-6);
}

#[test]
fn test_decay_without_decay_rate() {
    let mut interactions = UserItemInteractions::new(-5.0, 10.0, None); // No decay rate
    let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f32;

    interactions.add_interaction(1, 100, current_time, 5.0); // User 1, Item 100, delta 5.0
    assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 5.0);

    // Sleep for 2 seconds, but should not affect the value since decay is not applied
    sleep(Duration::from_secs(2));

    assert_eq!(interactions.get_user_item_rating(1, 100, 0.0), 5.0); // Value should still be 5.0
}
