use crate::bbox::Bbox;
use crate::inference_model::{InferenceModel, Inference, get_bbox};
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


// #[post("/get_face")]
// pub async fn get_face_bbox_yolo(
//     loaded_model: web::Data<InferenceModel>,
//     form: MultipartForm<GetFaceRequest>,
//     req: HttpRequest,
// ) -> actix_web::Result<HttpResponse> {
//     let get_face_req = form.into_inner();
//     let temp_file = get_face_req.input;
//     let img = tempfile_to_dynimg(temp_file)?;
//     let t1 = std::time::Instant::now();
//     let bboxes = loaded_model.forward(&img, 0.1).unwrap();
//     println!("inference time {:?}", t1.elapsed());
//     let mut _res = match bboxes {
//         InferenceResult::FaceDetection(ayylmao) => ayylmao,
//         _ => unreachable!(),
//     };
//     if _res.len() > 0 {
//         let _a = sort_conf_bbox(&mut _res);
//         Ok(HttpResponse::Ok().json(GetFaceResponse {
//             data: FaceResponse::from_bbox_vec(&_a),
//             message: String::from("success"),
//         }))
//     } else {
//         Ok(HttpResponse::Ok().json(GetFaceResponseNone {
//             message: String::from("no detections were found, please try with a better image!"),
//         }))
//     }
// }
