#[cfg(test)]
mod tests {

    use rtrec::identifiers::{Identifier, SerializableValue};

    #[test]
    fn test_identify_integer_with_pass_through() {
        let mut identifier = Identifier::new("Test Identifier");

        let obj = SerializableValue::Integer(42);
        let id = identifier.identify(obj.clone()).unwrap();

        // Pass-through mode should return the integer directly as the ID
        assert_eq!(id, 42);
        assert_eq!(identifier.get_id(&obj).unwrap(), Some(42));
    }

    #[test]
    fn test_identify_text() {
        let mut identifier = Identifier::new("Test Identifier");

        let obj = SerializableValue::Text("Hello".to_string());
        let id = identifier.identify(obj.clone()).unwrap();

        // Non-integer objects should be assigned new IDs starting from 0
        assert_eq!(id, 0);
        assert_eq!(identifier.get_id(&obj).unwrap(), Some(0));
    }

    #[test]
    fn test_identify_multiple_objects() {
        let mut identifier = Identifier::new("Test Identifier");

        let obj1 = SerializableValue::Text("Object 1".to_string());
        let obj2 = SerializableValue::Text("Object 2".to_string());
        let id1 = identifier.identify(obj1.clone()).unwrap();
        let id2 = identifier.identify(obj2.clone()).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_get_object_by_id() {
        let mut identifier = Identifier::new("Test Identifier");

        let obj = SerializableValue::Integer(99);
        let id = identifier.identify(obj.clone()).unwrap();
        let retrieved_obj = identifier.get(id).unwrap();

        assert_eq!(retrieved_obj, obj);
    }

    #[test]
    fn test_get_object_by_invalid_id() {
        let identifier = Identifier::new("Test Identifier");
        let result = identifier.get(999); // Invalid ID

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Identifier not found for Test Identifier: 999"
        );
    }

    #[test]
    fn test_get_or_default() {
        let mut identifier = Identifier::new("Test Identifier");

        let default_obj = SerializableValue::Text("Default".to_string());
        let obj = SerializableValue::Text("Object".to_string());
        let id = identifier.identify(obj.clone()).unwrap();

        assert_eq!(identifier.get_or_default(id, Some(default_obj.clone())), obj);

        // passthrough mode
        assert_eq!(identifier.get_or_default(999, Some(default_obj.clone())), default_obj);
    }

    #[test]
    fn test_get_id_nonexistent() {
        let identifier = Identifier::new("Test Identifier");

        let obj = SerializableValue::Text("Not Found".to_string());
        let id = identifier.get_id(&obj).unwrap();
        assert_eq!(id, None);
    }

    #[test]
    fn test_identify_text_then_integer_fails() {
        let mut identifier = Identifier::new("Test Identifier");

        let obj1 = SerializableValue::Text("Hello".to_string());
        let obj2 = SerializableValue::Integer(5);

        // Identify text first
        let id1 = identifier.identify(obj1.clone()).unwrap();
        assert_eq!(id1, 0);

        // Identify integer next
        let result2 = identifier.identify(obj2.clone());
        assert!(result2.is_err());

        let error = result2.unwrap_err();
        assert_eq!(error.to_string(), "Mixed types detected for Test Identifier: Integer(5)");
    }

    #[test]
    fn test_identify_integer_then_text_fails() {
        let mut identifier = Identifier::new("Test Identifier");

        let obj1 = SerializableValue::Integer(5);
        let obj2 = SerializableValue::Text("Hello".to_string());

        // Identify text first
        let id1 = identifier.identify(obj1.clone()).unwrap();
        assert_eq!(id1, 5);

        // Identify integer next
        let result2 = identifier.identify(obj2.clone());
        assert!(result2.is_err());

        let error = result2.unwrap_err();
        assert_eq!(error.to_string(), "Mixed types detected for Test Identifier: Text(\"Hello\")");
    }

}
