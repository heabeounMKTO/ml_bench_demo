use crate::bbox::Bbox;
use crate::inference_model::{get_bbox, Inference, InferenceModel};
use crate::web_ops::handler::{GetFaceRequest, GetFaceResponse};
use crate::web_ops::tempfile_to_dynimg;
use actix_multipart::form::MultipartForm;
use actix_web::http::header::ContentType;
use actix_web::{get, post, web, HttpRequest, HttpResponse};

#[get("/")]
pub async fn index(req: HttpRequest) -> HttpResponse {
    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .insert_header(("X-Hdr", "sample"))
        .body("server is up :)")
}

#[post("/get_face")]
pub async fn get_face_bbox_yolo(
    loaded_model: web::Data<InferenceModel>,
    form: MultipartForm<GetFaceRequest>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    let get_face_req = form.into_inner();
    let temp_file = get_face_req.input;
    let img = tempfile_to_dynimg(temp_file)?;
    let t1 = std::time::Instant::now();
    let bbox = get_bbox(loaded_model.get_ref(), &img, 0.5, 0.5).unwrap();
    let inf_time: f32 = t1.elapsed().as_millis() as f32;

    Ok(HttpResponse::Ok().json(GetFaceResponse::send_with_inference_time(bbox, inf_time)))
}
