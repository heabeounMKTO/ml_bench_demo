use crate::bbox::Bbox;
use crate::inference_model::{Inference, InferenceModel, TorchModel};
use anyhow::{Error, Result};
use image::{DynamicImage, GenericImageView, ImageEncoder};
use tch::Device;
use tch::{self, vision::image as tch_image};
use std::f64;
use std::io::{Cursor, Write, Read};
use tch::{kind, IValue, Tensor};

// add device later , just use CPU for now.
impl Inference for TorchModel {
    fn load(model_path: &str, fp16: bool) -> Result<InferenceModel, Error> {
        let mut model = tch::CModule::load_on_device(model_path, tch::Device::Cpu).unwrap();
        let loaded = TorchModel {
            model: model,
            is_fp16: fp16,
        };
        Ok(InferenceModel::TorchInferenceModel(loaded))
    }

    fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Bbox>, Error> {
        let pp = preprocess_image(input_image)?;
        let pred = self.model.forward_ts(&[pp])?.to_device(tch::Device::Cpu); 
        println!("PRED {:?}", pred.get(0));

        let _res = post_process_tch_fwd(&pred.get(0), confidence_threshold,iou_threshold, input_image);
        let _d: Vec<Bbox> = vec![];
        Ok(_d)
    }
}

fn scale_wh(w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
    let r = (w1 / w0).min(h1 / h0);
    (r, (w0 * r).round(), (h0 * r).round())
}
fn preprocess_image(input_image: &DynamicImage) -> Result<tch::Tensor, Error> {
    let mut buffer = Vec::new();
    let (width, height) = input_image.dimensions();
    image::codecs::jpeg::JpegEncoder::new(&mut buffer).encode(input_image.as_bytes(), width, height, input_image.color().into());
    let preproc = tch::vision::image::load_and_resize_from_memory(&buffer, 320, 320)?
        .unsqueeze(0)
        .to_kind(tch::Kind::Float)
        .to_device(tch::Device::Cpu)
        .g_div_scalar(255.);
    Ok(preproc)
}


fn post_process_tch_fwd(pred: &tch::Tensor, conf_thresh: f32, iou_thresh: f32, input_image: &DynamicImage) -> Result<Vec<Bbox>, Error> {
    let (npreds, pred_size) = pred.size2().unwrap();
    println!("NPREDS {:?}", npreds);
    let (_, w_new, h_new) = scale_wh(
        input_image.width() as f32,
        input_image.height() as f32,
        320.0,
        320.0,
    );
    let bbox_vec: Vec<Bbox> = vec![];
    let _pred = pred.squeeze_dim(0);
    for index  in 0..pred_size {
        // println!("row {:?}", _pred.get(4).get(index));
        let confidence = _pred.get(4).get(index).double_value(&[]) as f32;
        if confidence >= conf_thresh {
            println!("CONFIDENCE {:?}", confidence);
        }
    }
    println!("BBOX VEC {:?}", bbox_vec);
    Ok(bbox_vec)
}
