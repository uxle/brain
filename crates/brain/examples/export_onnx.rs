use brain::prelude::*;
use brain_graph::{build_mlp_graph, optimize, OptLevel};
use brain_onnx::{check_model, create_mlp_zoo_model, export_onnx_bytes};
use brain_quantization::{quantize_tensor, QuantConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = SyntheticClassification::two_class_points(4);
    let batches = data.batches(4);
    let model = Sequential::new()
        .add(Linear::new(2, 4, true))
        .add(ReLU::new())
        .add(Linear::new(4, 2, true));

    let mut trainer = Trainer::builder().model(model).learning_rate(0.2).build()?;
    trainer.fit(&batches, 4)?;

    let state = trainer.state();
    let checkpoint_bytes = state.to_brain_bytes();
    let mut graph = build_mlp_graph(2, 4, 2);
    let report = optimize(&mut graph, OptLevel::O1)
        .map_err(|err| format!("graph optimization failed: {}", err))?;
    let onnx_model = create_mlp_zoo_model();
    let checker = check_model(&onnx_model)?;
    let onnx_bytes = export_onnx_bytes(&onnx_model)?;
    let quantized = quantize_tensor(&state.tensors[0].tensor, &QuantConfig::default())?;

    println!(
        "export_onnx: checkpoint_bytes={} graph_nodes={} optimized_changes={} onnx_valid={} onnx_bytes={} qparams_scale={:.6}",
        checkpoint_bytes.len(),
        graph.nodes.len(),
        report.passes_applied,
        checker.is_valid,
        onnx_bytes.len(),
        quantized.params.scales[0]
    );

    Ok(())
}
