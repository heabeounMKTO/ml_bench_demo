mod bbox;
mod get_face_onnx;
mod get_face_torch;
mod get_face_tract;
mod inference_model;
mod web_ops;
use actix_web::middleware::Logger;
use actix_web::{web, App, HttpServer};
use anyhow::{Error, Result};
use clap::Parser;
use inference_model::{get_bbox, Inference, InferenceModel, OnnxModel, TorchModel, TractModel};
use web_ops::service::{get_face_bbox_yolo, index};

#[derive(clap::Parser)]
struct CliArgs {
    /// backend index
    /// 0 -> onnx
    /// 1 -> torch
    /// 2 -> tract
    #[arg(long)]
    backend: i32,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=debug");
    let args = CliArgs::parse();
    let backend_index = args.backend;
    let load_model: InferenceModel = match backend_index {
        0 => OnnxModel::load("models/yolov8n_face.onnx", false).unwrap(),
        1 => TorchModel::load("models/yolov8n-face.torchscript", false).unwrap(),
        2 => TractModel::load("models/yolov8n_face.onnx", false).unwrap(),
        _ => {
            panic!("invalid backend index supplied!?")
        }
    };
    let bind_addr = format!("0.0.0.0:{}", 9995);

    let _wrap_detector = web::Data::new(load_model);
    HttpServer::new(move || {
        App::new()
            .service(index)
            .service(get_face_bbox_yolo)
            .app_data(_wrap_detector.clone())

            .wrap(Logger::default())
    })
    .client_request_timeout(std::time::Duration::from_secs(0))
    .keep_alive(None)
    .bind(&bind_addr)?
    .workers(8)
    .run()
    .await
}
