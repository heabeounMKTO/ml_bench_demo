use crate::bbox::Bbox;
use actix_multipart::form::tempfile::TempFile;
use actix_multipart::form::text::Text;
use actix_multipart::form::MultipartForm;
use actix_multipart::Multipart;

use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, MultipartForm)]
pub struct GetFaceRequest {
    pub input: TempFile,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetFaceResponse {
    pub data: Vec<Bbox>,
    pub message: String,
}

impl GetFaceResponse {
    pub fn send_with_inference_time(bbox: Vec<Bbox>, inf_time: f32) -> GetFaceResponse {
        GetFaceResponse {
            data: bbox,
            message: format!("success! \n time: {}ms", inf_time),
        }
    }
}
