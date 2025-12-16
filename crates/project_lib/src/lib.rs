pub const fn add(value_a: i64, value_b: i64) -> i64 {
    value_a + value_b
}

#[cfg(test)]
mod test {
    use crate::add;

    #[test]
    fn test_add() {
        assert_eq!(add(1, 2), 3);
    }
}
