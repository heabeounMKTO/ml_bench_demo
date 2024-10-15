mod bbox;
mod get_face_onnx;
mod get_face_torch;
mod get_face_tract;
mod inference_model;

use anyhow::{Error, Result};
use inference_model::{get_bbox, Inference, InferenceModel, OnnxModel, TorchModel, TractModel};

fn main() -> Result<(), Error> {
    let load_model: InferenceModel = TractModel::load("models/yolov8n_face.onnx", false)?;
    let load_model_onnx: InferenceModel = OnnxModel::load("models/yolov8n_face.onnx", false)?;
    let load_model_tch: InferenceModel =
        TorchModel::load("models/yolov8n-face.torchscript", false)?;
    let test_img = image::open("/home/hbdesk/photo_2024-06-21_10-00-29.jpg").unwrap();
    let _a = get_bbox(load_model, &test_img, 0.5, 0.5)?;
    let _b = get_bbox(load_model_onnx, &test_img, 0.5, 0.5)?;
    let _c = get_bbox(load_model_tch, &test_img, 0.5, 0.5)?;
    println!("_tract {:?}\n_onnx {:?}\n_tch {:?}", _a, _b, _c);
    Ok(())
}
