mod bbox;
mod get_face_onnx;
mod get_face_torch;
mod get_face_tract;
mod inference_model;
mod web_ops;

use clap::Parser;
use actix_web::middleware::Logger;
use actix_web::{web, App, HttpServer};
use anyhow::{Error, Result};
use inference_model::{get_bbox, Inference, InferenceModel, OnnxModel, TorchModel, TractModel};

// fn main() -> Result<(), Error> {
//     let load_model: InferenceModel = TractModel::load("models/yolov8n_face.onnx", false)?;
//     let load_model_tch: InferenceModel =
//         TorchModel::load("models/yolov8n-face.torchscript", false)?;
//     let load_model_onnx: InferenceModel = OnnxModel::load("models/yolov8n_face.onnx", false)?;
//     let test_img = image::open("/home/hbdesk/photo_2024-06-21_10-00-29.jpg").unwrap();

//     let _a = get_bbox(load_model, &test_img, 0.5, 0.5)?;
//     let _b = get_bbox(load_model_onnx, &test_img, 0.5, 0.5)?;
//     let _c = get_bbox(load_model_tch, &test_img, 0.5, 0.5)?;
//     println!("_tract: {:?}\n_onnx: {:?}\n_tch: {:?}", _a, _b, _c);
//     Ok(())
// }
//

#[derive(clap::Parser)]
struct CliArgs {
    /// backend index
    /// 0 -> onnx 
    /// 1 -> torch
    /// 2 -> tract
    #[arg(long)]
    backend: i32
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=debug");
    let args = CliArgs::parse();
    let backend_index = args.backend;
    let load_model = match backend_index {
        0 => {
            OnnxModel::load("models/yolov8n_face.onnx", false);
        },
        
        1 => {
            TorchModel::load("models/yolov8n-face.torchscript", false);
        },
        2 => {
            TractModel::load("models/yolov8n_face.onnx", false);
        },
        _ => {
            panic!("invalid backend index supplied!?") 
        }
    }; 
    let bind_addr = format!("0.0.0.0:{}", 9995);

    HttpServer::new(move || {
        let _wrap_detector = web::Data::new(
            load_model
        );
        App::new().app_data(_wrap_detector).wrap(Logger::default())    
    })    
    .client_request_timeout(std::time::Duration::from_secs(0))
    .keep_alive(None)
    .bind(&bind_addr)?
    .workers(1)
    .run()
    .await
}
