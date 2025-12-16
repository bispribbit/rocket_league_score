use parking_lot::Mutex;
use project_lib::add;

#[derive(Default, Copy, Clone)]
struct Abc;

fn main() {
    Mutex::new(Abc);
    println!("Hello, world! {}", add(2, 4));
}
