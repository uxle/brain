use brain_core::Tensor;
use brain_dataset::dataset::{tabular::TabularDataset, Dataset, Subset};
use brain_dataset::loaders::DataLoader;

#[test]
fn test_tabular_dataset_and_subset() {
    let features = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]);
    let labels = Tensor::from_slice(&[0.0, 1.0, 0.0, 1.0], vec![4]);

    let ds = TabularDataset::new(features, Some(labels));
    assert_eq!(ds.len(), 4);
    assert!(!ds.is_empty());

    let item0 = ds.get(0).expect("Item 0");
    assert_eq!(item0.data.to_vec(), vec![1.0, 2.0]);
    assert_eq!(item0.target.unwrap().to_vec(), vec![0.0]);

    // Subset test
    let subset = Subset::new(&ds, vec![1, 3]);
    assert_eq!(subset.len(), 2);
    let s0 = subset.get(0).expect("Subset item 0 -> real item 1");
    assert_eq!(s0.data.to_vec(), vec![3.0, 4.0]);
    assert_eq!(s0.target.unwrap().to_vec(), vec![1.0]);
}

#[test]
fn test_dataloader_batch_fetch() {
    let features = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]);

    let ds = TabularDataset::new(features, None);
    let loader = DataLoader::new(&ds, 2);

    let batch = loader.fetch_batch().expect("Batch");
    assert_eq!(batch.len(), 2);
    assert_eq!(batch.items[0].data.to_vec(), vec![10.0, 20.0]);
    assert_eq!(batch.items[1].data.to_vec(), vec![30.0, 40.0]);
}

#[test]
fn test_tensor_dataset_and_concat_dataset() {
    use brain_dataset::{ConcatDataset, TensorDataset};

    let t1_data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t1_labels = Tensor::from_slice(&[0.0, 1.0], vec![2, 1]);
    let ds1 = TensorDataset::new(t1_data, Some(t1_labels));

    let t2_data = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
    let t2_labels = Tensor::from_slice(&[2.0], vec![1, 1]);
    let ds2 = TensorDataset::new(t2_data, Some(t2_labels));

    let concat = ConcatDataset::new(vec![&ds1, &ds2]);
    assert_eq!(concat.len(), 3);

    let item0 = concat.get(0).unwrap();
    assert_eq!(item0.data.data(), &[1.0, 2.0]);
    assert_eq!(item0.target.unwrap().data(), &[0.0]);

    let item2 = concat.get(2).unwrap();
    assert_eq!(item2.data.data(), &[5.0, 6.0]);
    assert_eq!(item2.target.unwrap().data(), &[2.0]);
}

#[test]
fn test_dataset_k_fold_and_stratification() {
    use brain_dataset::{k_fold_split_indices, stratified_split_indices};

    let folds = k_fold_split_indices(20, 4);
    assert_eq!(folds.len(), 4);
    for (train, val) in folds {
        assert_eq!(train.len(), 15);
        assert_eq!(val.len(), 5);
    }

    let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let (train, test) = stratified_split_indices(&labels, 0.67);
    assert_eq!(train.len(), 6);
    assert_eq!(test.len(), 3);
}
