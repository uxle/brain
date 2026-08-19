use brain::core::Tensor;

fn main() {
    println!("Hello from Brain project!");
    let t = Tensor::ones(vec![2, 2]);
    println!("Created tensor: {:?}", t.shape());
}
