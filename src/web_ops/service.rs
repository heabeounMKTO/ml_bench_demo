use crate::bbox::Bbox;
use crate::inference_model::{InferenceModel, Inference, get_bbox};
use actix_multipart::form::MultipartForm;
use actix_web::http::header::ContentType;
use actix_web::{get, post, web, HttpRequest, HttpResponse};






