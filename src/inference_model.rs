use crate::bbox::Bbox;
use crate::get_face_onnx;
use crate::get_face_tract;
use anyhow::{Error, Result};
use image::DynamicImage;
use ort::Session;
use tch::CModule;
use tract_onnx::prelude::*;

pub struct OnnxModel {
    pub is_fp16: bool,
    pub model: Session,
}

pub struct TorchModel {
    pub is_fp16: bool,
    pub model: CModule,
}

pub struct TractModel {
    pub is_fp16: bool,
    pub model: SimplePlan<
        TypedFact,
        Box<dyn TypedOp>,
        tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
    >,
}

pub enum InferenceModel {
    TorchInferenceModel(TorchModel),
    TractInferenceModel(TractModel),
    OnnxInferenceModel(OnnxModel),
}

// loading and forward pass abstraction
pub trait Inference {
    fn load(model_path: &str, fp16: bool) -> Result<InferenceModel, Error>;
    fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Bbox>, Error>;
}

/// big ass abstraction layer my homies
pub fn get_bbox(
    loaded_model: InferenceModel,
    input_image: &DynamicImage,
    confidence_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Bbox>, Error> {
    let bboxes: Vec<Bbox> = match loaded_model {
        InferenceModel::TractInferenceModel(_model) => {
            let _res: Vec<Bbox> =
                _model.forward(input_image, confidence_threshold, iou_threshold)?;
            _res
        }
        InferenceModel::TorchInferenceModel(_model) => {
            let _res: Vec<Bbox> =
                _model.forward(input_image, confidence_threshold, iou_threshold)?;
            _res
        }
        InferenceModel::OnnxInferenceModel(_model) => {
            let _res: Vec<Bbox> =
                _model.forward(input_image, confidence_threshold, iou_threshold)?;
            _res
        }
    };
    Ok(bboxes)
}
