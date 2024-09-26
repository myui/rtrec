use rtrec::datasets::UserItemInteractions;

#[test]
#[should_panic(expected = "max_value should be greater than min_value")]
fn test_initialization_invalid_bounds() {
    UserItemInteractions::new(10.0, -5.0);
}

#[test]
fn test_add_interaction() {
    let mut interactions = UserItemInteractions::new(-5.0, 10.0);

    interactions.add_interaction(1, 100, 3.0); // User 1, Item 100, delta 3.0
    assert_eq!(interactions.get_user_item_count(1, 100), 3.0);

    interactions.add_interaction(1, 100, 5.0); // Modify interaction
    assert_eq!(interactions.get_user_item_count(1, 100), 8.0); // Expect 8.0

    interactions.add_interaction(1, 100, 5.0); // Clamping above max
    assert_eq!(interactions.get_user_item_count(1, 100), 10.0); // Should be clamped to max_value

    interactions.add_interaction(1, 100, -100.0); // Clamping above max
    assert_eq!(interactions.get_user_item_count(1, 100), -5.0); // Should be clamped to max_value
}

#[test]
fn test_get_all_items_for_user() {
    let mut interactions = UserItemInteractions::new(-5.0, 10.0);
    interactions.add_interaction(1, 100, 3.0);
    interactions.add_interaction(1, 101, 2.0);

    let items = interactions.get_all_items_for_user(1);
    assert_eq!(items.len(), 2);
    assert!(items.contains(&100));
    assert!(items.contains(&101));
}

#[test]
fn test_get_all_non_interacted_items() {
    let mut interactions = UserItemInteractions::new(-5.0, 10.0);
    interactions.add_interaction(1, 100, 3.0);
    interactions.add_interaction(1, 101, 2.0);

    let non_interacted_items = interactions.get_all_non_interacted_items(1);
    assert!(!non_interacted_items.contains(&100));
    assert!(!non_interacted_items.contains(&101));
}

#[test]
fn test_get_all_non_negative_items() {
    let mut interactions = UserItemInteractions::new(-5.0, 10.0);
    interactions.add_interaction(1, 100, 3.0); // User 1, Item 100, delta 3.0
    interactions.add_interaction(1, 101, -5.0); // User 1, Item 101, delta -5.0 (will clamp to min_value)

    let non_negative_items = interactions.get_all_non_negative_items(1);
    assert_eq!(non_negative_items.len(), 1);
    assert!(non_negative_items.contains(&100)); // Only item 100 should be non-negative
    assert!(!non_negative_items.contains(&101)); // Item 101 should not be included as its value is -5.0
}